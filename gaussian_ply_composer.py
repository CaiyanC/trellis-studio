from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable

import numpy as np
from plyfile import PlyData, PlyElement


@dataclass
class GaussianPly:
    vertex: np.ndarray  # structured array
    prop_names: List[str]


# 简单读缓存：避免同一PLY在多次调参时重复解压/解析
_PLY_CACHE: Dict[Tuple[str, int, int], GaussianPly] = {}
_MAX_PLY_CACHE = 4


def _cache_key(path: str) -> Tuple[str, int, int]:
    p = Path(path).resolve()
    st = p.stat()
    return (str(p), int(st.st_mtime_ns), int(st.st_size))


def _read_gaussian_ply(path: str) -> GaussianPly:
    key = _cache_key(path)
    cached = _PLY_CACHE.get(key)
    if cached is not None:
        return cached

    ply = PlyData.read(path)
    if "vertex" not in ply:
        raise ValueError(f"PLY缺少vertex元素: {path}")

    v = ply["vertex"].data
    props = [p.name for p in ply["vertex"].properties]

    # Compressed format produced by splat-transform: has element chunk + packed_* fields
    if ("packed_position" in props) and ("chunk" in ply):
        parsed = _decompress_splat_transform_ply(ply, path)
    else:
        parsed = GaussianPly(vertex=v, prop_names=props)

    _PLY_CACHE[key] = parsed
    if len(_PLY_CACHE) > _MAX_PLY_CACHE:
        _PLY_CACHE.pop(next(iter(_PLY_CACHE)))
    return parsed


def _decompress_splat_transform_ply(ply: PlyData, src_path: str) -> GaussianPly:
    """
    Convert splat-transform compressed PLY into a standard "vertex" structured array with:
    x,y,z,nx,ny,nz,f_dc_0..2,opacity,scale_0..2,rot_0..3

    The compressed file usually has:
    - element chunk: per-chunk min/max ranges
    - element vertex: packed_position/rotation/scale/color (uint32)
    - element sh: optional higher-order SH stored as bytes (ignored)
    """
    chunk = ply["chunk"].data
    vtx = ply["vertex"].data

    if "packed_position" not in vtx.dtype.names:
        raise ValueError(f"不支持的compressed PLY（缺少packed_position）: {src_path}")

    n = len(vtx)
    chunk_count = len(chunk)

    # Heuristic from splat-transform: vertices are grouped in fixed-size chunks (typically 256).
    # This matches many compressed datasets: chunk_count * 256 ~= vertex_count.
    chunk_size = 256
    chunk_id = (np.arange(n, dtype=np.int64) // chunk_size).astype(np.int64)
    if chunk_count > 0:
        chunk_id = np.clip(chunk_id, 0, chunk_count - 1)

    def _lerp(min_v: np.ndarray, max_v: np.ndarray, t: np.ndarray) -> np.ndarray:
        return min_v + (max_v - min_v) * t

    # Decode positions: 3x10-bit quantization within chunk bounds
    pp = vtx["packed_position"].astype(np.uint32)
    qx = (pp >> 0) & 1023
    qy = (pp >> 10) & 1023
    qz = (pp >> 20) & 1023
    tx = qx.astype(np.float32) / 1023.0
    ty = qy.astype(np.float32) / 1023.0
    tz = qz.astype(np.float32) / 1023.0

    min_x = _as_f32(chunk["min_x"])[chunk_id]
    min_y = _as_f32(chunk["min_y"])[chunk_id]
    min_z = _as_f32(chunk["min_z"])[chunk_id]
    max_x = _as_f32(chunk["max_x"])[chunk_id]
    max_y = _as_f32(chunk["max_y"])[chunk_id]
    max_z = _as_f32(chunk["max_z"])[chunk_id]
    x = _lerp(min_x, max_x, tx)
    y = _lerp(min_y, max_y, ty)
    z = _lerp(min_z, max_z, tz)

    # Decode scales: 3x10-bit quantization within chunk bounds
    ps = vtx["packed_scale"].astype(np.uint32)
    qsx = (ps >> 0) & 1023
    qsy = (ps >> 10) & 1023
    qsz = (ps >> 20) & 1023
    tsx = qsx.astype(np.float32) / 1023.0
    tsy = qsy.astype(np.float32) / 1023.0
    tsz = qsz.astype(np.float32) / 1023.0

    min_sx = _as_f32(chunk["min_scale_x"])[chunk_id]
    min_sy = _as_f32(chunk["min_scale_y"])[chunk_id]
    min_sz = _as_f32(chunk["min_scale_z"])[chunk_id]
    max_sx = _as_f32(chunk["max_scale_x"])[chunk_id]
    max_sy = _as_f32(chunk["max_scale_y"])[chunk_id]
    max_sz = _as_f32(chunk["max_scale_z"])[chunk_id]
    sx = _lerp(min_sx, max_sx, tsx)
    sy = _lerp(min_sy, max_sy, tsy)
    sz = _lerp(min_sz, max_sz, tsz)

    # Convert to log-scale like classic 3DGS PLY conventions (TRELLIS save_ply uses log)
    eps = 1e-8
    scale_0 = np.log(np.maximum(sx, eps)).astype(np.float32)
    scale_1 = np.log(np.maximum(sy, eps)).astype(np.float32)
    scale_2 = np.log(np.maximum(sz, eps)).astype(np.float32)

    # Decode color/opacity: 4x8-bit, rgb is per-chunk lerp, alpha is direct [0,1]
    pc = vtx["packed_color"].astype(np.uint32)
    cr = (pc >> 0) & 255
    cg = (pc >> 8) & 255
    cb = (pc >> 16) & 255
    ca = (pc >> 24) & 255

    tr = cr.astype(np.float32) / 255.0
    tg = cg.astype(np.float32) / 255.0
    tb = cb.astype(np.float32) / 255.0
    a = ca.astype(np.float32) / 255.0

    min_r = _as_f32(chunk["min_r"])[chunk_id]
    min_g = _as_f32(chunk["min_g"])[chunk_id]
    min_b = _as_f32(chunk["min_b"])[chunk_id]
    max_r = _as_f32(chunk["max_r"])[chunk_id]
    max_g = _as_f32(chunk["max_g"])[chunk_id]
    max_b = _as_f32(chunk["max_b"])[chunk_id]
    r = _lerp(min_r, max_r, tr)
    g = _lerp(min_g, max_g, tg)
    b = _lerp(min_b, max_b, tb)

    # Convert rgb to SH DC coefficient (approx): f_dc = rgb / SH_C0
    SH_C0 = np.float32(0.28209479177387814)
    f_dc_0 = (r / SH_C0).astype(np.float32)
    f_dc_1 = (g / SH_C0).astype(np.float32)
    f_dc_2 = (b / SH_C0).astype(np.float32)

    # Convert alpha to inverse_sigmoid(opacity) to match classic PLY conventions.
    a = np.clip(a, 1e-6, 1.0 - 1e-6)
    opacity = np.log(a / (1.0 - a)).astype(np.float32)

    # Decode rotation quaternion: 4x int8 normalized to [-1,1], stored in uint32 bytes
    pr = vtx["packed_rotation"].astype(np.uint32)
    rb = pr.view(np.uint8).reshape(-1, 4).view(np.int8).astype(np.float32)
    q = rb / 127.0
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
    rot_0 = q[:, 0].astype(np.float32)
    rot_1 = q[:, 1].astype(np.float32)
    rot_2 = q[:, 2].astype(np.float32)
    rot_3 = q[:, 3].astype(np.float32)

    # Build standard vertex table
    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
        ("f_dc_0", "f4"),
        ("f_dc_1", "f4"),
        ("f_dc_2", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"),
        ("scale_1", "f4"),
        ("scale_2", "f4"),
        ("rot_0", "f4"),
        ("rot_1", "f4"),
        ("rot_2", "f4"),
        ("rot_3", "f4"),
    ]
    out = np.empty(n, dtype=dtype)
    out["x"] = x
    out["y"] = y
    out["z"] = z
    out["nx"] = 0
    out["ny"] = 0
    out["nz"] = 0
    out["f_dc_0"] = f_dc_0
    out["f_dc_1"] = f_dc_1
    out["f_dc_2"] = f_dc_2
    out["opacity"] = opacity
    out["scale_0"] = scale_0
    out["scale_1"] = scale_1
    out["scale_2"] = scale_2
    out["rot_0"] = rot_0
    out["rot_1"] = rot_1
    out["rot_2"] = rot_2
    out["rot_3"] = rot_3

    prop_names = [d[0] for d in dtype]
    return GaussianPly(vertex=out, prop_names=prop_names)


def _as_f32(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.float32:
        return arr
    return arr.astype(np.float32, copy=False)


def _get_xyz(v: np.ndarray) -> np.ndarray:
    for k in ["x", "y", "z"]:
        if k not in v.dtype.names:
            raise ValueError(f"PLY缺少字段: {k}")
    return np.stack([_as_f32(v["x"]), _as_f32(v["y"]), _as_f32(v["z"])], axis=1)


def _set_xyz(v: np.ndarray, xyz: np.ndarray) -> None:
    v["x"] = xyz[:, 0].astype(v["x"].dtype, copy=False)
    v["y"] = xyz[:, 1].astype(v["y"].dtype, copy=False)
    v["z"] = xyz[:, 2].astype(v["z"].dtype, copy=False)


def _quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Hamilton product for quaternions in wxyz order.
    q = q1 * q2
    """
    w1, x1, y1, z1 = q1.T
    w2, x2, y2, z2 = q2.T
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.stack([w, x, y, z], axis=1)


def _quat_from_euler_xyz_deg(rx: float, ry: float, rz: float) -> np.ndarray:
    """
    Create a quaternion (wxyz) from XYZ euler degrees.
    """
    # Convert to radians
    rx = math.radians(rx)
    ry = math.radians(ry)
    rz = math.radians(rz)

    cx, sx = math.cos(rx / 2), math.sin(rx / 2)
    cy, sy = math.cos(ry / 2), math.sin(ry / 2)
    cz, sz = math.cos(rz / 2), math.sin(rz / 2)

    # q = qx * qy * qz (XYZ intrinsic)
    qx = np.array([cx, sx, 0.0, 0.0], dtype=np.float32)
    qy = np.array([cy, 0.0, sy, 0.0], dtype=np.float32)
    qz = np.array([cz, 0.0, 0.0, sz], dtype=np.float32)

    q = _quat_mul_wxyz(qx[None, :], qy[None, :])
    q = _quat_mul_wxyz(q, qz[None, :])
    return q[0]


def _rotate_xyz_by_quat_wxyz(xyz: np.ndarray, q_wxyz: np.ndarray) -> np.ndarray:
    """
    Rotate xyz by quaternion (wxyz). Uses q * v * q^-1.
    """
    q = q_wxyz.astype(np.float32)
    q = q / (np.linalg.norm(q) + 1e-8)
    w, x, y, z = q

    # Rotation matrix from quaternion
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )
    return (xyz @ R.T).astype(np.float32)


def _ensure_props(v: np.ndarray, target_props: List[str]) -> np.ndarray:
    """
    Return a new structured array with exactly target_props in order.
    Missing props are filled with zeros.
    """
    dtype = []
    for p in target_props:
        if p in v.dtype.names:
            dtype.append((p, v.dtype[p]))
        else:
            dtype.append((p, np.float32))

    out = np.empty(len(v), dtype=dtype)
    for p in target_props:
        if p in v.dtype.names:
            out[p] = v[p]
        else:
            out[p] = np.zeros(len(v), dtype=out.dtype[p])
    return out


def _downsample_vertex(v: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0 or len(v) <= max_points:
        return v
    # 均匀抽样：比随机抽样更稳定，便于重复调试
    idx = np.linspace(0, len(v) - 1, num=max_points, dtype=np.int64)
    return v[idx]


def create_lightweight_ply(input_ply_path: str, out_ply_path: str, max_points: int = 350000) -> str:
    src = Path(input_ply_path)
    dst = Path(out_ply_path)
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        try:
            if dst.stat().st_mtime_ns >= src.stat().st_mtime_ns and dst.stat().st_size > 0:
                return str(dst)
        except Exception:
            pass

    g = _read_gaussian_ply(str(src))
    sampled = _downsample_vertex(g.vertex, max_points=max_points)
    el = PlyElement.describe(sampled, "vertex")
    PlyData([el]).write(str(dst))
    return str(dst)


def compose_gaussian_scene_multi_ply(
    scene_ply_path: str,
    objects: List[dict],
    out_ply_path: str,
    progress_cb: Optional[Callable[[float, str], None]] = None,
    preview_out_ply_path: Optional[str] = None,
    preview_max_points: int = 250000,
) -> str:
    """
    多物体合成：场景3DGS(.ply) + 多个物体3DGS(.ply) -> 输出合成ply。

    objects 每项包含：
    - path: str
    - position: (px, py, pz)
    - rotation_xyz_deg: (rx, ry, rz)
    - scale: float
    """
    valid_objects = [o for o in objects if o and o.get("path")]
    if not valid_objects:
        raise ValueError("至少需要一个物体PLY")

    def _progress(ratio: float, desc: str):
        if progress_cb is not None:
            progress_cb(max(0.0, min(1.0, float(ratio))), desc)

    _progress(0.05, "读取场景PLY")
    scene = _read_gaussian_ply(scene_ply_path)

    # 先统计所有属性并读取对象
    obj_parsed: List[Tuple[dict, GaussianPly]] = []
    union_props = list(scene.prop_names)
    read_start = 0.10
    read_span = 0.28
    for i, obj in enumerate(valid_objects):
        _progress(read_start + read_span * (i / max(1, len(valid_objects))), f"读取物体PLY {i+1}/{len(valid_objects)}")
        g = _read_gaussian_ply(obj["path"])
        obj_parsed.append((obj, g))
        for p in g.prop_names:
            if p not in union_props:
                union_props.append(p)

    _progress(0.42, "对齐属性")
    scene_v = _ensure_props(scene.vertex, union_props)
    composed_parts = [scene_v]

    transform_start = 0.45
    transform_span = 0.35
    for i, (obj_cfg, g) in enumerate(obj_parsed):
        _progress(transform_start + transform_span * (i / max(1, len(obj_parsed))), f"应用位姿 {i+1}/{len(obj_parsed)}")
        obj_v = _ensure_props(g.vertex, union_props)

        pos = tuple(obj_cfg.get("position", (0.0, 0.0, 0.0)))
        rot = tuple(obj_cfg.get("rotation_xyz_deg", (0.0, 0.0, 0.0)))
        sc = float(obj_cfg.get("scale", 1.0))

        xyz = _get_xyz(obj_v)
        xyz = xyz * sc
        q_pose = _quat_from_euler_xyz_deg(*rot)
        xyz = _rotate_xyz_by_quat_wxyz(xyz, q_pose)
        xyz = xyz + np.array(pos, dtype=np.float32)[None, :]
        _set_xyz(obj_v, xyz)

        rot_keys = [f"rot_{j}" for j in range(4)]
        if all(k in obj_v.dtype.names for k in rot_keys):
            q_obj = np.stack([_as_f32(obj_v[k]) for k in rot_keys], axis=1)
            q_pose_batch = np.repeat(q_pose[None, :], len(q_obj), axis=0)
            q_new = _quat_mul_wxyz(q_pose_batch, q_obj)
            q_new = q_new / (np.linalg.norm(q_new, axis=1, keepdims=True) + 1e-8)
            for j, k in enumerate(rot_keys):
                obj_v[k] = q_new[:, j].astype(obj_v.dtype[k], copy=False)

        if sc > 0:
            log_s = float(math.log(sc))
            for k in obj_v.dtype.names:
                if k.startswith("scale_"):
                    obj_v[k] = (_as_f32(obj_v[k]) + log_s).astype(obj_v.dtype[k], copy=False)

        composed_parts.append(obj_v)

    _progress(0.82, "合并高斯点云")
    composed_v = np.concatenate(composed_parts, axis=0)

    out_p = Path(out_ply_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    _progress(0.90, "写出完整合成PLY")
    el = PlyElement.describe(composed_v, "vertex")
    PlyData([el]).write(str(out_p))

    if preview_out_ply_path:
        _progress(0.95, "生成轻量预览PLY")
        preview_p = Path(preview_out_ply_path)
        preview_p.parent.mkdir(parents=True, exist_ok=True)
        sampled = _downsample_vertex(composed_v, max_points=preview_max_points)
        preview_el = PlyElement.describe(sampled, "vertex")
        PlyData([preview_el]).write(str(preview_p))

    _progress(1.0, "多物体合成完成")
    return str(out_p)

def compose_gaussian_scene_ply(
    scene_ply_path: str,
    object_ply_path: str,
    out_ply_path: str,
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    rotation_xyz_deg: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    scale: float = 1.0,
    progress_cb: Optional[Callable[[float, str], None]] = None,
    preview_out_ply_path: Optional[str] = None,
    preview_max_points: int = 250000,
) -> str:
    """向后兼容的单物体合成封装。"""
    return compose_gaussian_scene_multi_ply(
        scene_ply_path=scene_ply_path,
        objects=[
            {
                "path": object_ply_path,
                "position": position,
                "rotation_xyz_deg": rotation_xyz_deg,
                "scale": scale,
            }
        ],
        out_ply_path=out_ply_path,
        progress_cb=progress_cb,
        preview_out_ply_path=preview_out_ply_path,
        preview_max_points=preview_max_points,
    )


