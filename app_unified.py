import gradio as gr
from gradio_litmodel3d import LitModel3D

import json
import os
import shutil
from typing import *
import torch
import numpy as np
import imageio
from easydict import EasyDict as edict
from PIL import Image
from pathlib import Path
from trellis.pipelines import TrellisImageTo3DPipeline, TrellisTextTo3DPipeline
from trellis.representations import Gaussian, MeshExtractResult
from trellis.utils import render_utils, postprocessing_utils
import threading
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import unquote, urlencode
import re
from datetime import datetime
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# 导入场景编辑器
from scene_editor import SceneEditor
 
from object_placer import ObjectPlacer
from gaussian_ply_composer import compose_gaussian_scene_ply, compose_gaussian_scene_multi_ply


MAX_SEED = np.iinfo(np.int32).max
TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tmp')
os.makedirs(TMP_DIR, exist_ok=True)
GENERATED_ASSETS_DIR = Path(os.path.dirname(os.path.abspath(__file__))) / "generated_assets"
GENERATED_ASSETS_DIR.mkdir(exist_ok=True)


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _sanitize_asset_name(name: str) -> str:
    name = re.sub(r"\s+", "_", name.strip())
    name = re.sub(r"[^0-9A-Za-z_\-\u4e00-\u9fff]", "_", name)
    name = re.sub(r"_+", "_", name).strip("._")
    return name


def _validate_asset_name(name: str) -> str:
    asset_name = _sanitize_asset_name(name)
    if not asset_name:
        raise gr.Error("请先取名，再生成3D资产。")
    return asset_name


def _make_asset_key(asset_name: str) -> str:
    base = _sanitize_asset_name(asset_name)
    if not base:
        base = "asset"
    key = base
    idx = 2
    while (GENERATED_ASSETS_DIR / key).exists():
        key = f"{base}_{idx}"
        idx += 1
    return key


def _asset_dir(asset_key: str) -> Path:
    return GENERATED_ASSETS_DIR / asset_key


def _asset_metadata_path(asset_key: str) -> Path:
    return _asset_dir(asset_key) / "metadata.json"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> Optional[dict]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _save_preview_frame(video_path: str, out_path: Path) -> None:
    try:
        frames = imageio.mimread(video_path, memtest=False)
        if frames:
            imageio.imwrite(out_path, frames[0])
    except Exception:
        pass


def _export_glb_from_state_to_path(state: dict, out_path: Path, mesh_simplify: float = 0.95, texture_size: int = 1024) -> str:
    gs, mesh = unpack_state(state)
    glb = postprocessing_utils.to_glb(gs, mesh, simplify=mesh_simplify, texture_size=texture_size, verbose=False)
    glb.export(str(out_path))
    return str(out_path)


def _export_gaussian_from_state_to_path(state: dict, out_path: Path) -> str:
    gs, _ = unpack_state(state)
    gs.save_ply(str(out_path))
    return str(out_path)


def _create_asset_record(asset_name: str, input_type: str, state: dict, preview_video_path: str) -> dict:
    asset_key = _make_asset_key(asset_name)
    asset_dir = _asset_dir(asset_key)
    asset_dir.mkdir(parents=True, exist_ok=True)

    video_dst = asset_dir / "preview.mp4"
    glb_dst = asset_dir / "asset.glb"
    ply_dst = asset_dir / "asset.ply"
    poster_dst = asset_dir / "poster.png"

    shutil.copy2(preview_video_path, video_dst)
    _export_glb_from_state_to_path(state, glb_dst)
    _export_gaussian_from_state_to_path(state, ply_dst)
    _save_preview_frame(str(video_dst), poster_dst)

    metadata = {
        "asset_key": asset_key,
        "asset_name": asset_name,
        "input_type": input_type,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        "video_path": str(video_dst),
        "glb_path": str(glb_dst),
        "ply_path": str(ply_dst),
        "poster_path": str(poster_dst) if poster_dst.exists() else "",
    }
    _write_json(_asset_metadata_path(asset_key), metadata)
    return metadata


def _list_asset_records() -> List[dict]:
    records: List[dict] = []
    if not GENERATED_ASSETS_DIR.exists():
        return records
    for child in GENERATED_ASSETS_DIR.iterdir():
        if not child.is_dir():
            continue
        metadata = _read_json(child / "metadata.json")
        if not metadata:
            continue
        if not (metadata.get("video_path") and os.path.isfile(metadata["video_path"])):
            continue
        records.append(metadata)
    records.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
    return records


def _get_asset_record(asset_key: Optional[str]) -> Optional[dict]:
    if not asset_key:
        return None
    for record in _list_asset_records():
        if record.get("asset_key") == asset_key:
            return record
    return None


def _latest_asset_record() -> Optional[dict]:
    records = _list_asset_records()
    return records[0] if records else None


def _asset_choices(include_glb: bool = True, include_ply: bool = True) -> List[Tuple[str, str]]:
    choices = []
    for record in _list_asset_records():
        has_glb = bool(record.get("glb_path")) and os.path.isfile(record["glb_path"])
        has_ply = bool(record.get("ply_path")) and os.path.isfile(record["ply_path"])
        if include_glb and not has_glb and not include_ply:
            continue
        if include_ply and not has_ply and not include_glb:
            continue
        if include_glb and include_ply and not (has_glb or has_ply):
            continue
        label = f'{record["asset_name"]} [{record["input_type"]}]'
        choices.append((label, record["asset_key"]))
    return choices


def _first_ply_asset_key() -> Optional[str]:
    choices = _asset_choices(include_glb=False, include_ply=True)
    return choices[0][1] if choices else None


def _clear_generated_assets() -> Tuple[gr.Dropdown, gr.Dropdown, str, None, None, None, str]:
    if GENERATED_ASSETS_DIR.exists():
        shutil.rmtree(GENERATED_ASSETS_DIR)
    GENERATED_ASSETS_DIR.mkdir(exist_ok=True)
    return (
        gr.Dropdown(choices=[], value=None),
        gr.Dropdown(choices=[], value=None),
        "✅ 已清空后台生成资产（未删除场景文件）。",
        None,
        None,
        None,
        "",
    )

# --- three.js gaussian viewer server (no extra deps) ---
_VIEWER_PORT = int(os.environ.get("GS_VIEWER_PORT", "8010"))
_VIEWER_PORT_ACTUAL: int | None = None
_viewer_started = False
_viewer_lock = threading.Lock()


def _ensure_viewer_server():
    """
    启动一个本地静态服务：
    - /viewer/*  -> repo/gs_viewer/  (index.html)
    - /data/*    -> tmp/gs_viewer_data/ (会话生成/合成的 ply)
    """
    global _viewer_started, _VIEWER_PORT_ACTUAL
    if _viewer_started:
        return
    with _viewer_lock:
        if _viewer_started:
            return

        repo_root = os.path.dirname(os.path.abspath(__file__))
        viewer_root = os.path.join(repo_root, "gs_viewer")
        data_root = os.path.join(TMP_DIR, "gs_viewer_data")
        os.makedirs(data_root, exist_ok=True)

        class Handler(SimpleHTTPRequestHandler):
            def do_GET(self):
                # 移除查询参数进行路径匹配
                raw_path = self.path.split("?", 1)[0]
                path = unquote(raw_path)
                
                if path.startswith("/data/"):
                    rel = path[len("/data/"):]
                    self._serve_from(os.path.join(data_root, rel))
                    return
                
                # 修复路径匹配逻辑：只要是 /viewer/ 开头或根目录访问
                if path.startswith("/viewer/") or path in ["/", "/index.html"]:
                    if path in ["/", "/index.html"]:
                        rel = "index.html"
                    else:
                        rel = path[len("/viewer/"):]
                    
                    # 如果 rel 是空的（说明访问的是 /viewer/），默认指向 index.html
                    if not rel or rel == "/":
                        rel = "index.html"
                        
                    target = os.path.join(viewer_root, rel.lstrip("/"))
                    self._serve_from(target)
                    return

                # 其他请求（如 favicon.ico）直接 404
                self.send_response(404)
                self.end_headers()

            def _serve_from(self, file_path: str):
                file_path = os.path.abspath(file_path)
                if not os.path.isfile(file_path):
                    # print(f"[GS Viewer] 404 Not Found: {file_path}")
                    self.send_response(404)
                    self.end_headers()
                    return

                # print(f"[GS Viewer] Serving: {file_path}")
                # 根据后缀设置 Content-Type
                if file_path.endswith(".html"):
                    ctype = "text/html; charset=utf-8"
                elif file_path.endswith(".js"):
                    ctype = "text/javascript; charset=utf-8"
                elif file_path.endswith(".ply"):
                    ctype = "application/octet-stream"
                else:
                    ctype = "application/octet-stream"

                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                with open(file_path, "rb") as f:
                    shutil.copyfileobj(f, self.wfile)

            def log_message(self, format, *args):
                return

        # Try preferred port first; if already in use, fall back to an ephemeral port.
        try:
            httpd = ThreadingHTTPServer(("0.0.0.0", _VIEWER_PORT), Handler)
        except OSError as e:
            # errno 98: Address already in use
            if getattr(e, "errno", None) == 98:
                httpd = ThreadingHTTPServer(("0.0.0.0", 0), Handler)
            else:
                raise
        t = threading.Thread(target=httpd.serve_forever, daemon=True)
        t.start()
        _VIEWER_PORT_ACTUAL = int(httpd.server_address[1])
        _viewer_started = True


def _get_request_host(req: gr.Request) -> str:
    """从 Gradio 请求中提取主机名"""
    try:
        # 优先从 headers 里的 host 获取
        return req.request.headers.get("host", "127.0.0.1").split(":")[0]
    except:
        return "127.0.0.1"


def _viewer_url_for_ply(session_hash: str, ply_path: str, req: gr.Request) -> str:
    _ensure_viewer_server()
    data_root = os.path.join(TMP_DIR, "gs_viewer_data", session_hash)
    os.makedirs(data_root, exist_ok=True)
    dst = os.path.join(data_root, f"asset_{os.path.basename(ply_path)}")
    if os.path.abspath(ply_path) != os.path.abspath(dst):
        shutil.copy2(ply_path, dst)
    ply_url = f"/data/{session_hash}/{os.path.basename(dst)}"
    port = _VIEWER_PORT_ACTUAL or _VIEWER_PORT
    host = _get_request_host(req)
    query = urlencode({"ply": ply_url, "ts": str(int(os.path.getmtime(dst)))})
    return f"http://{host}:{port}/viewer/index.html?{query}"


def _viewer_url_for_editor(
    session_hash: str,
    scene_p: str,
    obj_p: str,
    req: gr.Request,
    pose: Optional[dict] = None,
    objects: Optional[list[dict]] = None,
) -> str:
    """
    为场景编辑器生成 URL，同时加载场景和物体。
    - 兼容单物体参数：obj + pose
    - 支持多物体参数：objects=[{path, position, rotation_xyz_deg, scale}, ...]
    """
    _ensure_viewer_server()

    def _prepare_file(p, prefix):
        if not p or not os.path.isfile(p):
            return None
        data_root = os.path.join(TMP_DIR, "gs_viewer_data", session_hash)
        os.makedirs(data_root, exist_ok=True)
        dst = os.path.join(data_root, f"{prefix}_{os.path.basename(p)}")
        if os.path.abspath(p) != os.path.abspath(dst):
            shutil.copy2(p, dst)
        return f"/data/{session_hash}/{os.path.basename(dst)}"

    s_url = _prepare_file(scene_p, "scene")
    port = _VIEWER_PORT_ACTUAL or _VIEWER_PORT
    host = _get_request_host(req)
    params = {}
    if s_url:
        params["scene"] = s_url

    prepared_objects = []
    for i, it in enumerate(objects or []):
        if not isinstance(it, dict):
            continue
        p = it.get("path")
        o_url = _prepare_file(p, f"object{i}")
        if not o_url:
            continue
        pos = it.get("position", [0, 0, 0])
        rot = it.get("rotation_xyz_deg", [0, 0, 0])
        sc = float(it.get("scale", 1.0))
        prepared_objects.append({
            "obj": o_url,
            "px": float(pos[0]),
            "py": float(pos[1]),
            "pz": float(pos[2]),
            "rx": float(rot[0]),
            "ry": float(rot[1]),
            "rz": float(rot[2]),
            "sc": sc,
        })

    # 多物体优先；否则走单物体兼容路径
    if prepared_objects:
        params["objects"] = json.dumps(prepared_objects, ensure_ascii=False, separators=(",", ":"))
    else:
        o_url = _prepare_file(obj_p, "object")
        if o_url:
            params["obj"] = o_url
        if pose:
            for key in ("px", "py", "pz", "rx", "ry", "rz", "sc"):
                if key in pose and pose[key] is not None:
                    params[key] = str(pose[key])

    params["ts"] = str(int(datetime.now().timestamp()))
    return f"http://{host}:{port}/viewer/index.html?{urlencode(params)}"


def start_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    os.makedirs(user_dir, exist_ok=True)


def end_session(req: gr.Request):
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    shutil.rmtree(user_dir)


def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Preprocess the input image.

    Args:
        image (Image.Image): The input image.

    Returns:
        Image.Image: The preprocessed image.
    """
    processed_image = image_pipeline.preprocess_image(image)
    return processed_image


def preprocess_images(images: List[Tuple[Image.Image, str]]) -> List[Image.Image]:
    """
    Preprocess a list of input images.

    Args:
        images (List[Tuple[Image.Image, str]]): The input images.

    Returns:
        List[Image.Image]: The preprocessed images.
    """
    images = [image[0] for image in images]
    processed_images = [image_pipeline.preprocess_image(image) for image in images]
    return processed_images


def pack_state(gs: Gaussian, mesh: MeshExtractResult) -> dict:
    return {
        'gaussian': {
            **gs.init_params,
            '_xyz': gs._xyz.cpu().numpy(),
            '_features_dc': gs._features_dc.cpu().numpy(),
            '_scaling': gs._scaling.cpu().numpy(),
            '_rotation': gs._rotation.cpu().numpy(),
            '_opacity': gs._opacity.cpu().numpy(),
        },
        'mesh': {
            'vertices': mesh.vertices.cpu().numpy(),
            'faces': mesh.faces.cpu().numpy(),
        },
    }


def unpack_state(state: dict) -> Tuple[Gaussian, edict, str]:
    gs = Gaussian(
        aabb=state['gaussian']['aabb'],
        sh_degree=state['gaussian']['sh_degree'],
        mininum_kernel_size=state['gaussian']['mininum_kernel_size'],
        scaling_bias=state['gaussian']['scaling_bias'],
        opacity_bias=state['gaussian']['opacity_bias'],
        scaling_activation=state['gaussian']['scaling_activation'],
    )
    gs._xyz = torch.tensor(state['gaussian']['_xyz'], device='cuda')
    gs._features_dc = torch.tensor(state['gaussian']['_features_dc'], device='cuda')
    gs._scaling = torch.tensor(state['gaussian']['_scaling'], device='cuda')
    gs._rotation = torch.tensor(state['gaussian']['_rotation'], device='cuda')
    gs._opacity = torch.tensor(state['gaussian']['_opacity'], device='cuda')

    mesh = edict(
        vertices=torch.tensor(state['mesh']['vertices'], device='cuda'),
        faces=torch.tensor(state['mesh']['faces'], device='cuda'),
    )

    return gs, mesh


def get_seed(randomize_seed: bool, seed: int) -> int:
    """
    Get the random seed.
    """
    return np.random.randint(0, MAX_SEED) if randomize_seed else seed


def generate_3d(
    input_type: str,
    asset_name: str,
    text_prompt: str,
    image: Image.Image,
    multiimages: List[Tuple[Image.Image, str]],
    is_multiimage: bool,
    seed: int,
    ss_guidance_strength: float,
    ss_sampling_steps: int,
    slat_guidance_strength: float,
    slat_sampling_steps: int,
    multiimage_algo: Literal["multidiffusion", "stochastic"],
    req: gr.Request,
    progress=gr.Progress(track_tqdm=False),
) -> Tuple[dict, str, str]:
    """
    Generate 3D model from either text or image input.

    Args:
        input_type (str): "text" or "image"
        text_prompt (str): The text prompt.
        image (Image.Image): The input image.
        multiimages (List[Tuple[Image.Image, str]]): The input images in multi-image mode.
        is_multiimage (bool): Whether is in multi-image mode.
        seed (int): The random seed.
        ss_guidance_strength (float): The guidance strength for sparse structure generation.
        ss_sampling_steps (int): The number of sampling steps for sparse structure generation.
        slat_guidance_strength (float): The guidance strength for structured latent generation.
        slat_sampling_steps (int): The number of sampling steps for structured latent generation.
        multiimage_algo (Literal["multidiffusion", "stochastic"]): The algorithm for multi-image generation.

    Returns:
        dict: The information of the generated 3D model.
        str: The path to the video of the 3D model.
    """
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    asset_name = _validate_asset_name(asset_name)

    def _progress(ratio: float, desc: str):
        progress(max(0.0, min(1.0, float(ratio))), desc=desc)

    _progress(0.02, "准备生成任务")

    if input_type == "text":
        # --- 本地最稳定翻译逻辑 ---
        if re.search(r'[\u4e00-\u9fff]', text_prompt):
            print(f"检测到中文提示词: {text_prompt}")
            if translator is not None:
                try:
                    # 使用预加载好的 translator
                    result = translator(text_prompt, max_length=512)
                    translated_prompt = result[0]['translation_text']
                    print(f"本地翻译结果: {translated_prompt}")
                    text_prompt = translated_prompt
                    gr.Info(f"已通过本地模型将中文翻译为英文: {text_prompt}")
                except Exception as e:
                    print(f"翻译出错: {e}")
                    gr.Warning("本地翻译失败，请检查模型文件夹 models/translation_zh_en")
            else:
                gr.Warning("翻译模型未加载，将尝试直接使用原词生成。")
        # --- 翻译逻辑结束 ---

        _progress(0.10, "文字理解与翻译")
        outputs = text_pipeline.run(
            text_prompt,
            seed=seed,
            formats=["gaussian", "mesh"],
            sparse_structure_sampler_params={
                "steps": ss_sampling_steps,
                "cfg_strength": ss_guidance_strength,
            },
            slat_sampler_params={
                "steps": slat_sampling_steps,
                "cfg_strength": slat_guidance_strength,
            },
        )
    else:  # image input
        if not is_multiimage:
            _progress(0.10, "单图预处理与编码")
            outputs = image_pipeline.run(
                image,
                seed=seed,
                formats=["gaussian", "mesh"],
                preprocess_image=False,
                sparse_structure_sampler_params={
                    "steps": ss_sampling_steps,
                    "cfg_strength": ss_guidance_strength,
                },
                slat_sampler_params={
                    "steps": slat_sampling_steps,
                    "cfg_strength": slat_guidance_strength,
                },
            )
        else:
            _progress(0.10, "多图预处理与编码")
            outputs = image_pipeline.run_multi_image(
                [image[0] for image in multiimages],
                seed=seed,
                formats=["gaussian", "mesh"],
                preprocess_image=False,
                sparse_structure_sampler_params={
                    "steps": ss_sampling_steps,
                    "cfg_strength": ss_guidance_strength,
                },
                slat_sampler_params={
                    "steps": slat_sampling_steps,
                    "cfg_strength": slat_guidance_strength,
                },
                mode=multiimage_algo,
            )

    # Import render_utils here to avoid dependency issues at startup
    _progress(0.72, "生成完成，渲染预览视频")
    from trellis.utils import render_utils
    video = render_utils.render_video(outputs['gaussian'][0], num_frames=120)['color']
    video_geo = render_utils.render_video(outputs['mesh'][0], num_frames=120)['normal']
    video = [np.concatenate([video[i], video_geo[i]], axis=1) for i in range(len(video))]
    video_path = os.path.join(user_dir, 'sample.mp4')
    imageio.mimsave(video_path, video, fps=15)
    _progress(0.88, "写入资产文件")
    state = pack_state(outputs['gaussian'][0], outputs['mesh'][0])
    metadata = _create_asset_record(asset_name, input_type, state, video_path)
    _progress(1.0, "完成")
    torch.cuda.empty_cache()
    return state, metadata["video_path"], metadata["asset_key"]


def extract_glb(
    state: dict,
    mesh_simplify: float,
    texture_size: int,
    req: gr.Request,
) -> Tuple[str, str]:
    """
    Extract a GLB file from the 3D model.

    Args:
        state (dict): The state of the generated 3D model.
        mesh_simplify (float): The mesh simplification factor.
        texture_size (int): The texture resolution.

    Returns:
        str: The path to the extracted GLB file.
    """
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    gs, mesh = unpack_state(state)
    # Import postprocessing_utils here to avoid dependency issues at startup
    from trellis.utils import postprocessing_utils
    glb = postprocessing_utils.to_glb(gs, mesh, simplify=mesh_simplify, texture_size=texture_size, verbose=False)
    glb_path = os.path.join(user_dir, 'sample.glb')
    glb.export(glb_path)
    torch.cuda.empty_cache()
    return glb_path, glb_path


def extract_gaussian(state: dict, req: gr.Request) -> Tuple[str, str]:
    """
    Extract a Gaussian file from the 3D model.

    Args:
        state (dict): The state of the generated 3D model.

    Returns:
        str: The path to the extracted Gaussian file.
    """
    user_dir = os.path.join(TMP_DIR, str(req.session_hash))
    gs, _ = unpack_state(state)
    gaussian_path = os.path.join(user_dir, 'sample.ply')
    gs.save_ply(gaussian_path)
    torch.cuda.empty_cache()
    return gaussian_path, gaussian_path


def _refresh_latest_asset_view() -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str]]:
    record = _latest_asset_record()
    if not record:
        return None, None, None, None, None
    return (
        record.get("asset_key"),
        record.get("video_path"),
        record.get("glb_path"),
        record.get("ply_path"),
        record.get("poster_path") or None,
    )


def _refresh_asset_dropdown() -> gr.Dropdown:
    choices = _asset_choices(include_glb=True, include_ply=True)
    value = choices[0][1] if choices else None
    return gr.Dropdown(choices=choices, value=value)


def _load_asset_by_key(asset_key: Optional[str]) -> Tuple[Optional[str], Optional[str], Optional[str], str]:
    record = _get_asset_record(asset_key) if asset_key else _latest_asset_record()
    if not record:
        return None, None, None, ""
    return (
        record.get("video_path"),
        record.get("glb_path"),
        record.get("ply_path"),
        record.get("asset_name", ""),
    )


def _asset_status_text(asset_key: Optional[str]) -> str:
    record = _get_asset_record(asset_key) if asset_key else _latest_asset_record()
    if not record:
        return "当前没有已入库的生成资产。"
    return (
        f'当前资产：{record["asset_name"]} | '
        f'类型：{record["input_type"]} | '
        f'时间：{record["updated_at"]}'
    )


def _select_asset_for_preview(asset_key: Optional[str]) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str], Optional[str], str, str, Optional[str], gr.Dropdown]:
    video_path, glb_path, ply_path, _ = _load_asset_by_key(asset_key)
    preview_path = glb_path or ply_path
    return (
        asset_key,
        video_path,
        preview_path,
        glb_path,
        ply_path,
        _asset_status_text(asset_key),
        ply_path or "",
        preview_path,
        gr.Dropdown(value=asset_key),
    )


def _refresh_all_asset_views() -> Tuple[gr.Dropdown, gr.Dropdown, str, Optional[str], Optional[str], Optional[str], Optional[str], str]:
    choices = _asset_choices(include_glb=True, include_ply=True)
    value = choices[0][1] if choices else None
    video_path, glb_path, ply_path, _ = _load_asset_by_key(value)
    preview_path = glb_path or ply_path
    ply_choices = _asset_choices(include_glb=False, include_ply=True)
    ply_value = value if value else (ply_choices[0][1] if ply_choices else None)
    status = _asset_status_text(value)
    return (
        gr.Dropdown(choices=choices, value=value),
        gr.Dropdown(choices=ply_choices, value=ply_value),
        status,
        value,
        video_path,
        preview_path,
        preview_path,
        _get_asset_record(ply_value).get("ply_path", "") if _get_asset_record(ply_value) else "",
    )


def prepare_multi_example() -> List[Image.Image]:
    multi_case = list(set([i.split('_')[0] for i in os.listdir("assets/example_multi_image")]))
    images = []
    for case in multi_case:
        _images = []
        for i in range(1, 4):
            img = Image.open(f'assets/example_multi_image/{case}_{i}.png')
            W, H = img.size
            img = img.resize((int(W / H * 512), 512))
            _images.append(np.array(img))
        images.append(Image.fromarray(np.concatenate(_images, axis=1)))
    return images


def split_image(image: Image.Image) -> List[Image.Image]:
    """
    Split an image into multiple views.
    """
    image = np.array(image)
    alpha = image[..., 3]
    alpha = np.any(alpha>0, axis=0)
    start_pos = np.where(~alpha[:-1] & alpha[1:])[0].tolist()
    end_pos = np.where(alpha[:-1] & ~alpha[1:])[0].tolist()
    images = []
    for s, e in zip(start_pos, end_pos):
        images.append(Image.fromarray(image[:, s:e+1]))
    return [preprocess_image(image) for image in images]


with gr.Blocks(
    title="TRELLIS 3D生成器",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="sky"),
    css="""
    :root {
        --bg: #f4f8fc;
        --surface: #ffffff;
        --surface-soft: #f7fbff;
        --surface-muted: #eef4fb;
        --border: #dbe7f3;
        --border-strong: #c6d7ea;
        --text: #10233e;
        --text-soft: #60758f;
        --primary: #2f6bff;
        --primary-strong: #1957f2;
        --primary-soft: #eaf2ff;
        --success: #e8f7ef;
        --danger: #fff1f1;
        --warning: #fff7e8;
        --radius-lg: 20px;
        --radius-md: 14px;
        --radius-sm: 10px;
        --shadow-sm: 0 8px 24px rgba(33, 78, 146, 0.06);
        --shadow-md: 0 18px 48px rgba(33, 78, 146, 0.10);
    }

    body, .gradio-container {
        background: linear-gradient(180deg, #f8fbff 0%, #f3f7fc 100%) !important;
        color: var(--text) !important;
    }

    .gradio-container {
        max-width: 1380px !important;
        padding: 24px 28px 40px !important;
    }

    .app-shell {
        gap: 0 !important;
        align-items: stretch !important;
    }

    .app-content {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-lg) !important;
        box-shadow: var(--shadow-sm) !important;
        padding: 24px !important;
    }

    .top-hero {
        background: linear-gradient(135deg, rgba(47,107,255,0.13), rgba(124,178,255,0.10) 45%, rgba(255,255,255,0.98));
        border: 1px solid var(--border);
        border-radius: 24px;
        box-shadow: 0 26px 60px rgba(33, 78, 146, 0.14);
        padding: 30px 34px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }

    .top-hero::after {
        content: "";
        position: absolute;
        right: -40px;
        top: -30px;
        width: 220px;
        height: 220px;
        background: radial-gradient(circle, rgba(47,107,255,0.18) 0%, rgba(47,107,255,0.04) 45%, transparent 70%);
        pointer-events: none;
    }

    .top-hero__row {
        display: flex;
        justify-content: space-between;
        gap: 24px;
        align-items: flex-start;
        flex-wrap: wrap;
        position: relative;
        z-index: 1;
    }

    .top-hero__title {
        margin: 0;
        font-size: 2.35rem;
        line-height: 1.06;
        font-weight: 800;
        color: var(--text);
        letter-spacing: -0.035em;
    }

    .hero-metrics {
        display: grid;
        grid-template-columns: repeat(3, minmax(120px, 1fr));
        gap: 12px;
        margin-top: 22px;
        max-width: 720px;
    }

    .hero-metric {
        background: rgba(255,255,255,0.82);
        border: 1px solid rgba(219,231,243,0.95);
        border-radius: 16px;
        padding: 14px 16px;
        box-shadow: 0 10px 22px rgba(33,78,146,0.08);
    }

    .hero-metric__label {
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-soft);
        margin-bottom: 8px;
        font-weight: 700;
    }

    .hero-metric__value {
        font-size: 1.05rem;
        font-weight: 800;
        color: var(--text);
    }

    .top-hero__status {
        min-width: 250px;
        background: rgba(255,255,255,0.92);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 16px 18px;
        box-shadow: 0 18px 34px rgba(33,78,146,0.08);
    }

    .top-hero__subtitle {
        margin: 10px 0 0;
        color: var(--text-soft);
        font-size: 1.05rem;
        line-height: 1.7;
        max-width: 720px;
        font-weight: 600;
    }

    .top-hero__badges {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        margin-top: 18px;
    }

    .panel-heading {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        margin-bottom: 14px;
    }

    .panel-heading__eyebrow {
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--primary-strong);
        font-weight: 700;
        margin-bottom: 6px;
    }

    .panel-heading__title {
        margin: 0;
        font-size: 1.08rem;
        font-weight: 800;
        color: var(--text);
        letter-spacing: -0.01em;
    }

    .panel-heading__desc {
        margin: 4px 0 0;
        color: var(--text-soft);
        font-size: 0.92rem;
        line-height: 1.65;
    }

    .surface-card {
        background: linear-gradient(180deg, #ffffff 0%, #f9fbff 100%) !important;
        border: 1px solid var(--border) !important;
        border-radius: 18px !important;
        box-shadow: var(--shadow-sm) !important;
        padding: 18px !important;
    }

    .toolbar-row {
        display: grid;
        gap: 14px;
    }

    .asset-panel .wrap,
    .result-panel .wrap {
        gap: 14px !important;
    }

    .result-panel {
        position: relative;
        overflow: hidden;
    }

    .result-panel::after {
        content: "";
        position: absolute;
        inset: auto -40px -60px auto;
        width: 180px;
        height: 180px;
        background: radial-gradient(circle, rgba(47,107,255,0.12) 0%, rgba(47,107,255,0.02) 55%, transparent 72%);
        pointer-events: none;
    }

    .muted-block {
        background: #f8fbff;
        border: 1px dashed #d4e3f7;
        border-radius: 14px;
        padding: 12px 14px;
    }

    .muted-block p {
        margin: 0;
        color: var(--text-soft);
        line-height: 1.65;
        font-size: 0.92rem;
    }

    .workflow-card {
        background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 16px;
        box-shadow: var(--shadow-sm);
    }

    .workflow-card + .workflow-card {
        margin-top: 14px;
    }

    .step-switcher {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 10px;
        margin: 14px 0 16px;
    }

    .step-switcher button {
        min-height: 52px !important;
        font-weight: 700 !important;
        border-radius: 14px !important;
        background: #f8fbff !important;
        color: var(--text-soft) !important;
        border: 1px solid var(--border) !important;
    }

    .step-switcher button.step-active {
        background: linear-gradient(135deg, var(--primary-soft), #f4f8ff) !important;
        color: var(--primary-strong) !important;
        border: 1px solid #cfe0ff !important;
        box-shadow: 0 12px 24px rgba(47,107,255,0.12) !important;
    }

    .step-panel {
        margin-top: 6px;
    }

    .step-panel .wrap {
        gap: 12px !important;
    }

    .step-section-sep {
        height: 1px;
        background: linear-gradient(90deg, rgba(210,225,244,0), rgba(210,225,244,1), rgba(210,225,244,0));
        margin: 10px 0 4px;
    }

    .workflow-step {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        min-width: 30px;
        height: 30px;
        padding: 0 10px;
        border-radius: 999px;
        background: var(--primary-soft);
        color: var(--primary-strong);
        font-size: 0.82rem;
        font-weight: 800;
        margin-bottom: 10px;
    }

    .workflow-title {
        margin: 0 0 8px 0;
        color: var(--text);
        font-size: 1rem;
        font-weight: 800;
        letter-spacing: -0.01em;
    }

    .workflow-note {
        margin: 0 0 12px 0;
        color: var(--text-soft);
        line-height: 1.65;
        font-size: 0.92rem;
    }

    .status-box textarea,
    .status-box input {
        background: #f8fbff !important;
    }

    .action-row {
        margin-top: 8px;
    }

    .top-hero__badge {
        background: rgba(255,255,255,0.86);
        border: 1px solid var(--border);
        color: var(--text);
        padding: 8px 14px;
        border-radius: 999px;
        font-size: 0.93rem;
        font-weight: 600;
        backdrop-filter: blur(10px);
    }

    .top-hero__status {
        min-width: 220px;
        background: rgba(255,255,255,0.9);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 14px 16px;
    }

    .top-hero__status small {
        display: block;
        color: var(--text-soft);
        font-size: 0.78rem;
        margin-bottom: 6px;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .top-hero__status strong {
        color: var(--primary-strong);
        font-size: 1rem;
    }

    .section-title {
        margin: 0 0 10px 0;
        font-size: 1.04rem;
        font-weight: 800;
        color: var(--text);
        letter-spacing: -0.01em;
    }

    .section-note {
        color: var(--text-soft);
        font-size: 0.93rem;
        line-height: 1.7;
        margin: 0;
    }

    .compact-card {
        background: linear-gradient(180deg, #ffffff 0%, #f9fbff 100%) !important;
        border: 1px solid var(--border) !important;
        border-radius: 16px !important;
        box-shadow: var(--shadow-sm) !important;
        padding: 14px !important;
    }

    .tab-nav button,
    .tab-button {
        font-size: 15px !important;
        font-weight: 700 !important;
        border-radius: 14px !important;
        min-height: 46px !important;
        background: #f8fbff !important;
        border: 1px solid transparent !important;
        color: var(--text-soft) !important;
    }

    .tab-nav {
        gap: 10px !important;
        border-bottom: 1px solid var(--border) !important;
        padding-bottom: 14px !important;
        margin-bottom: 18px !important;
    }

    .tab-nav button.selected,
    .tab-button.selected {
        background: var(--primary-soft) !important;
        color: var(--primary-strong) !important;
        border: 1px solid #cfe0ff !important;
        box-shadow: inset 0 0 0 1px rgba(47,107,255,0.05), 0 8px 18px rgba(47,107,255,0.10) !important;
    }

    .gradio-dropdown ul li:nth-child(even),
    .gradio-dataframe table tbody tr:nth-child(even) {
        background: #f8fbff !important;
    }

    .input-section {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        padding: 0 !important;
        margin: 0 !important;
    }

    .control-panel,
    .output-section,
    .block.padded,
    .gr-group {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-md) !important;
        box-shadow: var(--shadow-sm) !important;
    }

    .control-panel,
    .output-section {
        padding: 18px !important;
        margin: 14px 0 !important;
    }

    .output-section {
        background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%) !important;
        color: var(--text) !important;
    }

    .generate-btn {
        background: linear-gradient(135deg, var(--primary) 0%, #4d8dff 100%) !important;
        color: white !important;
        font-size: 17px !important;
        font-weight: 700 !important;
        min-height: 54px !important;
        padding: 12px 24px !important;
        border-radius: 14px !important;
        border: none !important;
        box-shadow: 0 14px 30px rgba(47, 107, 255, 0.24) !important;
        transition: all 0.2s ease !important;
    }

    .generate-btn:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 18px 34px rgba(47, 107, 255, 0.34) !important;
    }

    button:not(.generate-btn), .gr-button-secondary, .gr-button {
        border-radius: 12px !important;
    }

    .gr-button-secondary {
        background: #ffffff !important;
        color: var(--primary-strong) !important;
        border: 1px solid #cfe0ff !important;
    }

    .gr-button-secondary:hover {
        background: var(--primary-soft) !important;
    }

    .gr-button-stop {
        background: #fff5f5 !important;
        color: #c53030 !important;
        border: 1px solid #ffd6d6 !important;
    }

    input, textarea, select {
        border-radius: 12px !important;
        border: 1px solid var(--border-strong) !important;
        background: #ffffff !important;
        color: var(--text) !important;
    }

    textarea {
        line-height: 1.7 !important;
    }

    input:focus, textarea:focus, select:focus {
        border-color: #8ab0ff !important;
        box-shadow: 0 0 0 4px rgba(47, 107, 255, 0.10) !important;
    }

    label, .gr-form label, .gr-block label {
        color: var(--text) !important;
        font-weight: 600 !important;
    }

    .prose h3, .prose h4, .prose p, .gr-markdown {
        color: var(--text) !important;
    }

    .gr-markdown p, .gr-markdown li, .gr-html p {
        color: var(--text-soft) !important;
        line-height: 1.65 !important;
    }

    .gr-accordion {
        border-radius: var(--radius-md) !important;
        overflow: hidden !important;
        border: 1px solid var(--border) !important;
        box-shadow: var(--shadow-sm) !important;
    }

    .gr-accordion .label-wrap {
        background: linear-gradient(180deg, #ffffff 0%, #f7fbff 100%) !important;
    }

    .gradio-container .gr-box,
    .gradio-container .gr-panel {
        border-radius: var(--radius-md) !important;
    }

    @media (max-width: 980px) {
        .gradio-container {
            padding: 16px !important;
        }

        .top-hero {
            padding: 20px 18px;
        }

        .top-hero__title {
            font-size: 1.8rem;
        }
    }
    """
) as demo:
    gr.HTML("""
    <section class="top-hero">
        <div class="top-hero__row">
            <div>
                <h1 class="top-hero__title">TRELLIS Studio</h1>
                <p class="top-hero__subtitle">3D资产工作台</p>
                <div class="top-hero__badges">
                    <span class="top-hero__badge">📝 文本生成 3D</span>
                    <span class="top-hero__badge">🖼️ 图片生成 3D</span>
                    <span class="top-hero__badge">📷 多视角融合</span>
                    <span class="top-hero__badge">🏗️ 场景编辑与导出</span>
                </div>
                <div class="hero-metrics">
                    <div class="hero-metric">
                        <div class="hero-metric__label">Workflow</div>
                        <div class="hero-metric__value">生成 · 导出 · 编辑</div>
                    </div>
                    <div class="hero-metric">
                        <div class="hero-metric__label">Assets</div>
                        <div class="hero-metric__value">GLB / Gaussian PLY</div>
                    </div>
                    <div class="hero-metric">
                        <div class="hero-metric__label">Modes</div>
                        <div class="hero-metric__value">文本 / 单图 / 多视角</div>
                    </div>
                </div>
            </div>
            <div class="top-hero__status">
                <small>Workspace</small>
                <strong>TRELLIS 3D Pipeline</strong>
                <div style="margin-top:10px;color:#60758f;font-size:0.92rem;line-height:1.7;">
                    面向展示与生产使用的 3D 资产生成工作台，统一生成、预览、导出与场景编辑流程。
                </div>
            </div>
        </div>
    </section>
    """)

    # 跨Tab共享：最近导出的物体文件路径（避免切Tab后丢失）
    # 必须在任何引用它们的事件绑定之前定义
    latest_object_glb = gr.State()
    latest_object_ply = gr.State()
    current_asset_key = gr.State()

    with gr.Row(elem_classes=["app-shell"]):
        with gr.Column(scale=1, elem_classes=["app-content"]):
            with gr.Tabs(elem_classes=["input-section"]) as input_tabs:
                with gr.Tab(label="📝 文字输入", id=0) as text_input_tab:
                    gr.HTML("<div class='panel-heading'><div><div class='panel-heading__eyebrow'>Input</div><h3 class='panel-heading__title'>文本生成</h3><p class='panel-heading__desc'>输入描述后直接生成 3D 资产。</p></div></div>")
                    text_prompt = gr.Textbox(
                        label="描述您想要生成的3D对象",
                        lines=5,
                        placeholder="例如：一个可爱的卡通猫坐在垫子上、一辆红色的跑车在赛道上、一把现代客厅的木椅..."
                    )

                with gr.Tab(label="🖼️ 单图片", id=1) as single_image_input_tab:
                    gr.HTML("<div class='panel-heading'><div><div class='panel-heading__eyebrow'>Input</div><h3 class='panel-heading__title'>单图生成</h3><p class='panel-heading__desc'>上传参考图片，保留原有生成流程与输出能力。</p></div></div>")
                    image_prompt = gr.Image(
                        label="上传参考图片",
                        format="png",
                        image_mode="RGBA",
                        type="pil",
                        height=300
                    )

                with gr.Tab(label="📷 多视角", id=2) as multiimage_input_tab:
                    gr.HTML("<div class='panel-heading'><div><div class='panel-heading__eyebrow'>Input</div><h3 class='panel-heading__title'>多视角生成</h3><p class='panel-heading__desc'>上传多张视角图片进行融合生成。</p></div></div>")
                    multiimage_prompt = gr.Gallery(
                        label="上传多视角图片",
                        format="png",
                        type="pil",
                        height=300,
                        columns=3
                    )

                with gr.Group(visible=True) as asset_name_group:
                    gr.HTML("<div class='panel-heading'><div><div class='panel-heading__eyebrow'>Asset</div><h3 class='panel-heading__title'>资产信息</h3><p class='panel-heading__desc'>填写资产名称，生成后用于资产管理、导出与后续复用。</p></div></div>")
                    asset_name = gr.Textbox(
                        label="资产名称（必填）",
                        placeholder="例如：aoqi_chair / 客厅桌子 / train_lamp",
                    )

                generated_assets_status = gr.Textbox(label="资产状态", interactive=False)

                with gr.Tab(label="🏗️ 场景编辑器", id=3) as scene_editor_tab:

                    gr.HTML("<div class='compact-card' style='margin-bottom:14px;'><div class='panel-heading' style='margin-bottom:0;'><div><div class='panel-heading__eyebrow'>Workspace</div><h3 class='panel-heading__title'>场景编辑工作台</h3><p class='panel-heading__desc'>保留场景选择、物体管理、位姿调整、合成预览与进入交互编辑器的完整流程。</p></div></div></div>")

                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.HTML("<div class='panel-heading'><div><div class='panel-heading__eyebrow'>Scene</div><h3 class='panel-heading__title'>场景与物体配置</h3><p class='panel-heading__desc'>按照场景源、物体源、物体列表与位姿的顺序完成配置。</p></div></div>")
                            with gr.Row(elem_classes=['step-switcher']):
                                scene_step_btn = gr.Button("01 场景源", variant="secondary")
                                object_step_btn = gr.Button("02 物体源", variant="secondary")
                                list_step_btn = gr.Button("03 物体列表与位姿", variant="secondary")

                            current_scene_summary = gr.Textbox(label="当前场景文件", interactive=False)

                            with gr.Group(visible=True, elem_classes=['step-panel']) as scene_step_group:
                                gr.HTML("<div class='workflow-card'><div class='workflow-step'>01</div><h4 class='workflow-title'>场景源</h4><p class='workflow-note'>选择已有场景高斯或上传新的场景 PLY，用于当前场景合成。</p></div>")
                                gs_scene_id = gr.Textbox(label="场景ID（仅用于输出目录命名，不是文件选择）", value="interior_0001_839920")

                                def _list_local_scene_plys() -> list[str]:
                                    import glob
                                    paths = glob.glob("scenes_gs/**/*.ply", recursive=True)
                                    paths = [p for p in paths if os.path.isfile(p)]
                                    paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                                    return paths

                                gs_scene_dropdown = gr.Dropdown(
                                    label="从服务器已存在的场景高斯中选择（推荐）",
                                    choices=_list_local_scene_plys(),
                                    interactive=True,
                                )
                                refresh_scene_dropdown_btn = gr.Button("🔄 刷新场景列表", variant="secondary")

                                with gr.Row():
                                    delete_scene_dropdown = gr.Dropdown(
                                        label="选择要删除的场景文件",
                                        choices=_list_local_scene_plys(),
                                        interactive=True,
                                    )
                                    delete_scene_btn = gr.Button("🗑️ 删除选中场景", variant="stop")
                                delete_scene_status = gr.Textbox(label="删除状态", interactive=False)
                                gs_scene_upload = gr.File(
                                    label="选择本地场景高斯PLY（3dgs_compressed.ply）",
                                    file_types=[".ply"],
                                    file_count="single",
                                    type="filepath",
                                )
                                gr.HTML("<div class='info-banner' style='margin:10px 0 12px 0;'><strong>上传提示</strong>：选择本地场景文件后会先上传再处理，请耐心等待进度条完成，期间不要刷新页面或切换标签页。</div>")
                                gs_scene_path = gr.Textbox(label="已选择的场景PLY路径", interactive=False)
                                gr.HTML("<div class='muted-block'><p>场景 ID 仅用于输出目录命名；真正的场景文件以列表选择或上传结果为准。</p></div>")

                            with gr.Group(visible=False, elem_classes=['step-panel']) as object_step_group:
                                gr.HTML("<div class='workflow-card'><div class='workflow-step'>02</div><h4 class='workflow-title'>物体源</h4><p class='workflow-note'>从已生成资产中选择物体高斯，或上传新的物体 PLY。</p></div>")
                                def _list_local_object_plys() -> list[tuple[str, str]]:
                                    return _asset_choices(include_glb=False, include_ply=True)

                                obj_dropdown = gr.Dropdown(
                                    label="从服务器已生成物体中选择（推荐）",
                                    choices=_list_local_object_plys(),
                                    interactive=True,
                                )
                                refresh_obj_dropdown_btn = gr.Button("🔄 刷新物体列表", variant="secondary")
                                with gr.Row():
                                    delete_obj_dropdown = gr.Dropdown(
                                        label="选择要删除的物体资产",
                                        choices=_list_local_object_plys(),
                                        interactive=True,
                                    )
                                    delete_obj_btn = gr.Button("🗑️ 删除选中物体", variant="stop")
                                delete_obj_status = gr.Textbox(label="删除物体状态", interactive=False)

                                use_latest_ply_btn = gr.Button("⬅️ 使用当前选中资产的物体高斯", variant="secondary")
                                gs_object_upload = gr.File(
                                    label="选择本地物体PLY",
                                    file_types=[".ply"],
                                    file_count="single",
                                    type="filepath",
                                )
                                gr.HTML("<div class='info-banner' style='margin:10px 0 12px 0;'><strong>上传提示</strong>：选择本地物体文件后会先上传再处理，请耐心等待进度条完成，期间不要刷新页面或切换标签页。</div>")
                                gs_object_path = gr.Textbox(label="已选择的物体PLY路径", interactive=False)
                                gs_objects_state = gr.State([])
                                gs_selected_object_idx = gr.State(-1)

                            with gr.Group(visible=False, elem_classes=['step-panel']) as list_step_group:
                                gr.HTML("<div class='workflow-card'><div class='workflow-step'>03</div><h4 class='workflow-title'>物体列表与位姿</h4><p class='workflow-note'>先管理物体列表，再调整位姿参数，最后执行场景合成。</p></div>")
                                gr.HTML("<div class='panel-heading' style='margin-bottom:8px;'><div><h4 class='panel-heading__title' style='font-size:0.98rem;'>物体列表</h4></div></div>")
                                gs_objects_dropdown = gr.Dropdown(
                                    label="已添加物体列表（选择后可编辑该物体）",
                                    choices=[],
                                    interactive=True,
                                )
                                with gr.Row():
                                    gs_add_object_btn = gr.Button("➕ 添加当前物体", variant="secondary")
                                    gs_update_object_btn = gr.Button("💾 用当前参数更新选中物体", variant="secondary")
                                with gr.Row():
                                    gs_remove_object_btn = gr.Button("➖ 删除选中物体", variant="secondary")
                                    gs_clear_objects_btn = gr.Button("🧹 清空物体列表", variant="secondary")

                                gr.HTML("<div class='step-section-sep'></div>")
                                gr.HTML("<div class='panel-heading' style='margin-bottom:8px;'><div><h4 class='panel-heading__title' style='font-size:0.98rem;'>位姿参数</h4></div></div>")
                                with gr.Row():
                                    gs_pos_x = gr.Slider(-10, 10, value=0, step=0.05, label="X位置")
                                    gs_pos_y = gr.Slider(-10, 10, value=0, step=0.05, label="Y位置")
                                    gs_pos_z = gr.Slider(-10, 10, value=0, step=0.05, label="Z位置")
                                with gr.Row():
                                    gs_rot_x = gr.Slider(-180, 180, value=0, step=5, label="X旋转(度)")
                                    gs_rot_y = gr.Slider(-180, 180, value=0, step=5, label="Y旋转(度)")
                                    gs_rot_z = gr.Slider(-180, 180, value=0, step=5, label="Z旋转(度)")
                                gs_scale = gr.Slider(0.01, 20.0, value=1.0, step=0.05, label="缩放")

                                gr.HTML("<div class='step-section-sep'></div>")
                                gr.HTML("<div class='panel-heading' style='margin-bottom:8px;'><div><h4 class='panel-heading__title' style='font-size:0.98rem;'>执行合成</h4></div></div>")
                                gs_compose_btn = gr.Button("🧩 合成高斯场景（输出PLY）", variant="primary")
                                gs_status = gr.Textbox(label="状态", interactive=False)

                        with gr.Column(scale=1):
                            gr.HTML("""
                            <div class='compact-card'>
                                <h4 class='section-title'>操作视图</h4>
                                <p class='section-note'>右侧区域用于聚焦当前操作结果：包括当前资产预览、合成场景预览、下载结果以及进入交互编辑器。</p>
                            </div>
                            """)

                    def _step_button_updates(active_idx: int):
                        base = ""
                        active = "step-active"
                        return (
                            gr.Button(elem_classes=[active] if active_idx == 1 else [base]),
                            gr.Button(elem_classes=[active] if active_idx == 2 else [base]),
                            gr.Button(elem_classes=[active] if active_idx == 3 else [base]),
                        )

                    def _show_scene_step():
                        b1, b2, b3 = _step_button_updates(1)
                        return gr.Group(visible=True), gr.Group(visible=False), gr.Group(visible=False), b1, b2, b3

                    def _show_object_step():
                        b1, b2, b3 = _step_button_updates(2)
                        return gr.Group(visible=False), gr.Group(visible=True), gr.Group(visible=False), b1, b2, b3

                    def _show_list_step():
                        b1, b2, b3 = _step_button_updates(3)
                        return gr.Group(visible=False), gr.Group(visible=False), gr.Group(visible=True), b1, b2, b3

                    scene_step_btn.click(_show_scene_step, inputs=[], outputs=[scene_step_group, object_step_group, list_step_group, scene_step_btn, object_step_btn, list_step_btn])
                    object_step_btn.click(_show_object_step, inputs=[], outputs=[scene_step_group, object_step_group, list_step_group, scene_step_btn, object_step_btn, list_step_btn])
                    list_step_btn.click(_show_list_step, inputs=[], outputs=[scene_step_group, object_step_group, list_step_group, scene_step_btn, object_step_btn, list_step_btn])

                    def _file_to_path(f):
                        if not f:
                            return ""
                        if isinstance(f, str):
                            return f
                        if isinstance(f, dict):
                            return f.get("path") or f.get("name") or ""
                        return getattr(f, "name", None) or getattr(f, "path", None) or ""

                    def _set_scene_ply(file_obj, progress=gr.Progress(track_tqdm=False)):
                        # 上传后持久化保存到 scenes_gs/uploaded/ 目录，让下拉列表能选到
                        progress(0.05, desc="开始处理场景文件")
                        path = _file_to_path(file_obj)
                        if path and os.path.isfile(path):
                            progress(0.35, desc="准备保存到场景目录")
                            upload_dir = os.path.join("scenes_gs", "uploaded")
                            os.makedirs(upload_dir, exist_ok=True)
                            dst = os.path.join(upload_dir, os.path.basename(path))
                            if os.path.abspath(path) != os.path.abspath(dst):
                                shutil.copy2(path, dst)
                            progress(0.75, desc="刷新场景列表")
                            saved_path = dst
                            status = f"✅ 已上传并保存场景文件：{os.path.basename(saved_path)}（可在下拉列表中选择）"
                            new_paths = _list_local_scene_plys()
                            summary = f"已选择：{saved_path}"
                            b1, b2, b3 = _step_button_updates(1)
                            progress(1.0, desc="上传处理完成")
                            return saved_path, gr.Dropdown(choices=new_paths, value=saved_path), status, summary, gr.Group(visible=True), gr.Group(visible=False), gr.Group(visible=False), b1, b2, b3
                        else:
                            status = "⚠️ 本地场景文件选择失败，请重试。"
                            b1, b2, b3 = _step_button_updates(1)
                            progress(1.0, desc="上传失败")
                            return "", gr.Dropdown(value=None), status, "", gr.Group(visible=True), gr.Group(visible=False), gr.Group(visible=False), b1, b2, b3

                    def _set_object_ply(file_obj, progress=gr.Progress(track_tqdm=False)):
                        progress(0.1, desc="开始处理物体文件")
                        path = _file_to_path(file_obj)
                        b1, b2, b3 = _step_button_updates(3 if path else 2)
                        progress(1.0, desc="上传处理完成" if path else "上传失败")
                        return path, gr.Group(visible=False), gr.Group(visible=not bool(path)), gr.Group(visible=bool(path)), b1, b2, b3

                    def _use_latest_object_ply(asset_key: str):
                        return _asset_key_to_ply(asset_key)

                    def _asset_key_to_ply(asset_key: str) -> str:
                        record = _get_asset_record(asset_key)
                        if not record:
                            return ""
                        return record.get("ply_path", "")

                    def _asset_key_to_scene_selection(asset_key: str):
                        record = _get_asset_record(asset_key)
                        if not record:
                            return None, "", None, "当前没有已入库的生成资产。", gr.Dropdown(value=None)
                        preview_path = record.get("glb_path") or record.get("ply_path") or None
                        return (
                            record.get("asset_key"),
                            record.get("ply_path", ""),
                            preview_path,
                            _asset_status_text(asset_key),
                            gr.Dropdown(value=asset_key),
                        )

                    def _object_display_name(path: str) -> str:
                        path = str(path or "")
                        if not path:
                            return "unnamed-object"
                        base = os.path.basename(path)
                        parent = os.path.basename(os.path.dirname(path))
                        stem = os.path.splitext(base)[0]
                        if base in {"asset.ply", "asset.glb", "3dgs_compressed.ply", "composed_scene.ply", "composed_scene_preview.ply"} and parent:
                            return parent
                        return stem or base or parent or "unnamed-object"

                    def _objects_dropdown_update(items: list[dict], selected_idx: int = -1):
                        choices = []
                        for i, it in enumerate(items):
                            name = str(it.get("name") or _object_display_name(it.get("path", "")))
                            pos = it.get("position", [0, 0, 0])
                            choices.append((f"#{i+1} {name} @ ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})", str(i)))
                        value = str(selected_idx) if 0 <= selected_idx < len(items) else ("0" if choices else None)
                        return gr.Dropdown(choices=choices, value=value)

                    def _add_object_to_list(obj_p: str, items: list[dict], px, py, pz, rx, ry, rz, sc):
                        items = list(items or [])
                        if not obj_p:
                            return items, _objects_dropdown_update(items, -1), -1, "⚠️ 请先选择物体PLY再添加"
                        item = {
                            "path": obj_p,
                            "name": _object_display_name(obj_p),
                            "position": [float(px), float(py), float(pz)],
                            "rotation_xyz_deg": [float(rx), float(ry), float(rz)],
                            "scale": float(sc),
                        }
                        items.append(item)
                        idx = len(items) - 1
                        return items, _objects_dropdown_update(items, idx), idx, f"✅ 已添加物体 #{idx+1}"

                    def _select_object_from_list(idx_str: str, items: list[dict]):
                        items = list(items or [])
                        if idx_str is None or idx_str == "":
                            return -1, "", 0, 0, 0, 0, 0, 0, 1.0
                        idx = int(idx_str)
                        if idx < 0 or idx >= len(items):
                            return -1, "", 0, 0, 0, 0, 0, 0, 1.0
                        it = items[idx]
                        pos = it.get("position", [0, 0, 0])
                        rot = it.get("rotation_xyz_deg", [0, 0, 0])
                        scv = float(it.get("scale", 1.0))
                        return idx, str(it.get("path", "")), float(pos[0]), float(pos[1]), float(pos[2]), float(rot[0]), float(rot[1]), float(rot[2]), scv

                    def _update_selected_object(idx: int, items: list[dict], obj_p: str, px, py, pz, rx, ry, rz, sc):
                        items = list(items or [])
                        if idx is None or idx < 0 or idx >= len(items):
                            return items, _objects_dropdown_update(items, -1), "⚠️ 请先在列表中选择一个物体"
                        if not obj_p:
                            return items, _objects_dropdown_update(items, idx), "⚠️ 当前物体路径为空，无法更新"
                        items[idx] = {
                            "path": obj_p,
                            "name": _object_display_name(obj_p),
                            "position": [float(px), float(py), float(pz)],
                            "rotation_xyz_deg": [float(rx), float(ry), float(rz)],
                            "scale": float(sc),
                        }
                        return items, _objects_dropdown_update(items, idx), f"✅ 已更新物体 #{idx+1}"

                    def _remove_selected_object(idx: int, items: list[dict]):
                        items = list(items or [])
                        if idx is None or idx < 0 or idx >= len(items):
                            return items, _objects_dropdown_update(items, -1), -1, "⚠️ 请先选择要删除的物体"
                        items.pop(idx)
                        new_idx = min(idx, len(items) - 1)
                        if len(items) == 0:
                            new_idx = -1
                        return items, _objects_dropdown_update(items, new_idx), new_idx, f"✅ 已删除物体 #{idx+1}"

                    def _clear_object_list():
                        items = []
                        return items, _objects_dropdown_update(items, -1), -1, "✅ 已清空物体列表"

                    def _compose_gaussian(scene_id: str, scene_p: str, obj_p: str, items: list[dict], px, py, pz, rx, ry, rz, sc, req: gr.Request, progress=gr.Progress(track_tqdm=False)):
                        if not scene_p:
                            return "❌ 请先选择场景高斯PLY", " ", gr.DownloadButton(interactive=False), None

                        objects = list(items or [])

                        # 向后兼容：列表为空时使用当前单物体参数
                        if not objects and obj_p:
                            objects = [{
                                "path": obj_p,
                                "position": [float(px), float(py), float(pz)],
                                "rotation_xyz_deg": [float(rx), float(ry), float(rz)],
                                "scale": float(sc),
                            }]

                        if not objects:
                            return "❌ 请先添加至少一个物体", " ", gr.DownloadButton(interactive=False), None

                        out_dir = os.path.join("scenes_gs", scene_id)
                        os.makedirs(out_dir, exist_ok=True)
                        out_ply = os.path.join(out_dir, "composed_scene.ply")
                        preview_ply = os.path.join(out_dir, "composed_scene_preview.ply")
                        try:
                            progress(0.02, desc="准备多物体合成任务")
                            compose_gaussian_scene_multi_ply(
                                scene_ply_path=scene_p,
                                objects=objects,
                                out_ply_path=out_ply,
                                progress_cb=lambda ratio, desc: progress(ratio, desc=desc),
                                preview_out_ply_path=preview_ply,
                                preview_max_points=260000,
                            )
                            progress(0.97, desc="生成编辑器链接")
                            viewer_obj = objects[-1].get("path") if isinstance(objects[-1], dict) else obj_p
                            viewer_url = _viewer_url_for_editor(
                                str(req.session_hash),
                                scene_p,
                                viewer_obj,
                                req,
                                pose={
                                    "px": float(px),
                                    "py": float(py),
                                    "pz": float(pz),
                                    "rx": float(rx),
                                    "ry": float(ry),
                                    "rz": float(rz),
                                    "sc": float(sc),
                                },
                                objects=objects,
                            )
                            progress(1.0, desc="完成")
                            md = (
                                f"✅ 多物体合成完成：共 {len(objects)} 个物体。\n"
                                "⚡ 右侧预览使用轻量抽样PLY，加快显示速度。\n\n"
                                f"👉 [打开交互编辑器：在此移动物体并获取坐标]({viewer_url})"
                            )
                            preview_out = preview_ply if os.path.isfile(preview_ply) else out_ply
                            return f"✅ 合成成功：{out_ply}", md, gr.DownloadButton(value=out_ply, interactive=True), preview_out
                        except Exception as e:
                            return f"❌ 合成失败：{e}", " ", gr.DownloadButton(interactive=False), None


                    # 事件绑定：选择文件 -> 显示路径
                    gs_scene_upload.change(_set_scene_ply, inputs=[gs_scene_upload], outputs=[gs_scene_path, gs_scene_dropdown, gs_status, current_scene_summary, scene_step_group, object_step_group, list_step_group, scene_step_btn, object_step_btn, list_step_btn])
                    gs_object_upload.change(_set_object_ply, inputs=[gs_object_upload], outputs=[gs_object_path, scene_step_group, object_step_group, list_step_group, scene_step_btn, object_step_btn, list_step_btn])

                    def _pick_scene_from_dropdown(p: str):
                        picked = p or ""
                        status = f"✅ 已切换到服务器场景文件：{os.path.basename(picked)}" if picked else "⚠️ 未选择服务器场景文件。"
                        summary = f"已选择：{picked}" if picked else ""
                        b1, b2, b3 = _step_button_updates(2 if picked else 1)
                        return picked, status, summary, gr.Group(visible=not bool(picked)), gr.Group(visible=bool(picked)), gr.Group(visible=False), b1, b2, b3

                    def _refresh_scene_dropdown():
                        paths = _list_local_scene_plys()
                        return gr.Dropdown(choices=paths, value=paths[0] if paths else None)

                    gs_scene_dropdown.change(_pick_scene_from_dropdown, inputs=[gs_scene_dropdown], outputs=[gs_scene_path, gs_status, current_scene_summary, scene_step_group, object_step_group, list_step_group, scene_step_btn, object_step_btn, list_step_btn])
                    refresh_scene_dropdown_btn.click(_refresh_scene_dropdown, inputs=[], outputs=[gs_scene_dropdown])

                    gs_add_object_btn.click(
                        _add_object_to_list,
                        inputs=[gs_object_path, gs_objects_state, gs_pos_x, gs_pos_y, gs_pos_z, gs_rot_x, gs_rot_y, gs_rot_z, gs_scale],
                        outputs=[gs_objects_state, gs_objects_dropdown, gs_selected_object_idx, gs_status],
                    )

                    gs_objects_dropdown.change(
                        _select_object_from_list,
                        inputs=[gs_objects_dropdown, gs_objects_state],
                        outputs=[gs_selected_object_idx, gs_object_path, gs_pos_x, gs_pos_y, gs_pos_z, gs_rot_x, gs_rot_y, gs_rot_z, gs_scale],
                    )

                    gs_update_object_btn.click(
                        _update_selected_object,
                        inputs=[gs_selected_object_idx, gs_objects_state, gs_object_path, gs_pos_x, gs_pos_y, gs_pos_z, gs_rot_x, gs_rot_y, gs_rot_z, gs_scale],
                        outputs=[gs_objects_state, gs_objects_dropdown, gs_status],
                    )

                    gs_remove_object_btn.click(
                        _remove_selected_object,
                        inputs=[gs_selected_object_idx, gs_objects_state],
                        outputs=[gs_objects_state, gs_objects_dropdown, gs_selected_object_idx, gs_status],
                    )

                    gs_clear_objects_btn.click(
                        _clear_object_list,
                        inputs=[],
                        outputs=[gs_objects_state, gs_objects_dropdown, gs_selected_object_idx, gs_status],
                    )

                    def _pick_object_from_dropdown(asset_key: str):
                        picked = asset_key or ""
                        path = _asset_key_to_ply(picked) if picked else ""
                        b1, b2, b3 = _step_button_updates(3 if path else 2)
                        return path, gr.Group(visible=False), gr.Group(visible=not bool(path)), gr.Group(visible=bool(path)), b1, b2, b3

                    obj_dropdown.change(_pick_object_from_dropdown, inputs=[obj_dropdown], outputs=[gs_object_path, scene_step_group, object_step_group, list_step_group, scene_step_btn, object_step_btn, list_step_btn])

                    def _refresh_obj_dropdown():
                        choices = _list_local_object_plys()
                        value = choices[0][1] if choices else None
                        return gr.Dropdown(choices=choices, value=value)

                    refresh_obj_dropdown_btn.click(_refresh_obj_dropdown, inputs=[], outputs=[obj_dropdown])


                    def _delete_scene_ply(scene_path: str):
                        if not scene_path or not os.path.isfile(scene_path):
                            new_paths = _list_local_scene_plys()
                            return "⚠️ 请先选择要删除的场景文件", gr.Dropdown(choices=new_paths, value=None), gr.Dropdown(choices=new_paths, value=None)
                        try:
                            os.remove(scene_path)
                            new_paths = _list_local_scene_plys()
                            return f"✅ 已删除场景文件：{os.path.basename(scene_path)}", gr.Dropdown(choices=new_paths, value=None), gr.Dropdown(choices=new_paths, value=None)
                        except Exception as e:
                            new_paths = _list_local_scene_plys()
                            return f"❌ 删除失败：{e}", gr.Dropdown(choices=new_paths, value=None), gr.Dropdown(choices=new_paths, value=None)

                    def _delete_object_asset(asset_key: str):
                        if not asset_key:
                            choices = _list_local_object_plys()
                            return "⚠️ 请先选择要删除的物体资产", gr.Dropdown(choices=choices, value=None), gr.Dropdown(choices=_asset_choices(include_glb=True, include_ply=True), value=None)
                        asset_dir = _asset_dir(asset_key)
                        try:
                            if asset_dir.exists():
                                shutil.rmtree(str(asset_dir))
                            choices = _list_local_object_plys()
                            all_choices = _asset_choices(include_glb=True, include_ply=True)
                            return f"✅ 已删除物体资产：{asset_key}", gr.Dropdown(choices=choices, value=None), gr.Dropdown(choices=all_choices, value=None)
                        except Exception as e:
                            choices = _list_local_object_plys()
                            all_choices = _asset_choices(include_glb=True, include_ply=True)
                            return f"❌ 删除失败：{e}", gr.Dropdown(choices=choices, value=None), gr.Dropdown(choices=all_choices, value=None)

                    delete_scene_btn.click(_delete_scene_ply, inputs=[delete_scene_dropdown], outputs=[delete_scene_status, delete_scene_dropdown, gs_scene_dropdown])
                    delete_obj_btn.click(_delete_object_asset, inputs=[delete_obj_dropdown], outputs=[delete_obj_status, delete_obj_dropdown, obj_dropdown])
                    # 一键使用“最新导出的物体高斯”
                    use_latest_ply_btn.click(_use_latest_object_ply, inputs=[current_asset_key], outputs=[gs_object_path])
                    # 合成按钮绑定会在右侧栏组件创建完成后进行（见下方），以便输出到右侧栏预览/下载区。

            with gr.Accordion(label="🎛️ 生成参数设置", open=False, elem_classes=["control-panel"]) as gen_params_panel:
                seed = gr.Slider(0, MAX_SEED, label="随机种子", value=0, step=1)
                randomize_seed = gr.Checkbox(label="🎲 每次随机种子", value=True)
                gr.HTML("<h4 style='color: #667eea; margin: 15px 0 10px 0;'>🏗️ 第一阶段：稀疏结构生成</h4>")
                with gr.Row():
                    ss_guidance_strength = gr.Slider(0.0, 10.0, label="引导强度", value=7.5, step=0.1)
                    ss_sampling_steps = gr.Slider(1, 50, label="采样步数", value=20, step=1)
                gr.HTML("<h4 style='color: #764ba2; margin: 15px 0 10px 0;'>🔮 第二阶段：结构化潜在生成</h4>")
                with gr.Row():
                    slat_guidance_strength = gr.Slider(0.0, 10.0, label="引导强度", value=3.0, step=0.1)
                    slat_sampling_steps = gr.Slider(1, 50, label="采样步数", value=20, step=1)
                multiimage_algo = gr.Radio(
                    ["stochastic", "multidiffusion"],
                    label="多图片算法",
                    value="stochastic",
                    visible=False
                )

            generate_btn = gr.Button("🚀 开始生成", elem_classes=["generate-btn"])

            with gr.Accordion(label="⚙️ 导出设置", open=False, elem_classes=["control-panel"]) as export_panel:
                mesh_simplify = gr.Slider(0.9, 0.98, label="网格简化度", value=0.95, step=0.01)
                texture_size = gr.Slider(512, 2048, label="纹理分辨率", value=1024, step=512)

            with gr.Row():
                extract_glb_btn = gr.Button("📦 导出GLB", interactive=False, variant="secondary")
                extract_gs_btn = gr.Button("🔸 导出高斯", interactive=False, variant="secondary")
            with gr.Row():
                refresh_latest_asset_btn = gr.Button("🔄 刷新最新资产", variant="secondary")
                clear_generated_assets_btn = gr.Button("🗑️ 清空生成资产", variant="stop")
            gr.HTML("""
            <div class="info-banner" style="margin-top:12px;">
                <strong>文件说明</strong>：高斯文件通常较大（约数十 MB），显示与下载需要一定时间，属于正常现象。
            </div>
            """)

        with gr.Column(scale=1):
            # 右侧栏：生成模式输出
            with gr.Group(visible=True) as gen_output_group:
                gr.HTML("<div class='panel-heading'><div><div class='panel-heading__eyebrow'>Output</div><h3 class='panel-heading__title'>生成结果与资产管理</h3><p class='panel-heading__desc'>在此查看预览、检查模型并完成文件下载与后台资产选择。</p></div></div>")
                with gr.Group(elem_classes=["output-section", "compact-card", "result-panel"]):
                    gr.HTML("<h3 class='section-title' style='text-align:center;'>🎬 生成结果预览</h3>")
                    video_output = gr.Video(label="3D资产预览", autoplay=True, loop=True, height=350, show_label=False)

                with gr.Group(elem_classes=["output-section", "compact-card"]):
                    gr.HTML("<h3 class='section-title' style='text-align:center;'>📦 3D模型查看器</h3>")
                    model_output = LitModel3D(label="GLB/高斯模型查看器", exposure=10.0, height=350, show_label=False)

                gr.HTML("<h4 class='section-title' style='text-align:center; margin: 18px 0 8px 0;'>💾 下载文件</h4><p class='section-note' style='text-align:center;'>生成完成后可直接下载 GLB 与高斯文件，后台资产会同步更新。</p>")
                with gr.Row():
                    download_glb = gr.DownloadButton(label="⬇️ 下载GLB文件", interactive=False, variant="primary")
                    download_gs = gr.DownloadButton(label="⬇️ 下载高斯文件", interactive=False, variant="primary")
                asset_library_dropdown = gr.Dropdown(
                    label="后台资产列表",
                    choices=_asset_choices(include_glb=True, include_ply=True),
                    interactive=True,
                )
                gr.HTML("<div class='muted-block'><p>可在此快速切换后台资产记录，用于复查导出结果与后续场景复用。</p></div>")

            # 右侧栏：场景合成/编辑模式输出（你要的“在整个页面右边”）
            with gr.Group(visible=False) as scene_output_group:
                gr.HTML("<div class='panel-heading'><div><div class='panel-heading__eyebrow'>Output</div><h3 class='panel-heading__title'>场景合成结果</h3><p class='panel-heading__desc'>查看当前资产预览、合成后的高斯场景，并完成下载与跳转编辑。</p></div></div>")
                gr.HTML("<h3 style='text-align: center; margin: 0 0 15px 0;' class='section-title'>🏗️ 场景编辑器输出</h3><p class='section-note' style='text-align:center; margin-bottom:12px;'>保留当前物体预览、合成结果预览、下载与交互编辑入口。</p>")
                scene_asset_view = LitModel3D(label="当前选择资产预览", exposure=10.0, height=320)
                refresh_scene_assets_btn = gr.Button("🔄 刷新场景资产", variant="secondary")
                gr.HTML("<h4 style='text-align: center; margin: 10px 0 8px 0; color: #666;'>🧩 合成高斯</h4>")
                composed_ply_view = LitModel3D(label="合成后的高斯PLY预览", exposure=10.0, height=320)
                refresh_composed_preview_btn = gr.Button("🔄 刷新合成预览", variant="secondary")
                gs_download_right = gr.DownloadButton(label="⬇️ 下载合成PLY", interactive=False, variant="primary")
                gs_viewer_link_right = gr.Markdown("（合成后会出现交互预览链接）")

    is_multiimage = gr.State(False)
    input_type = gr.State("text")
    output_buf = gr.State()

    # Example images at the bottom of the page
    with gr.Row(elem_classes=["control-panel"]) as text_example:
        gr.HTML("<h3 style='text-align: center; margin: 0 0 15px 0;'>💡 文字示例提示词</h3>")
        text_examples = gr.Examples(
            examples=[
                "A cute cartoon cat sitting on a cushion",
                "A red sports car on a racetrack",
                "A wooden chair in a modern living room",
                "A blue coffee mug on a wooden table",
                "A vintage steam locomotive train",
            ],
            inputs=[text_prompt],
            examples_per_page=5,
            label="点击选择示例",
        )

    with gr.Row(visible=False) as single_image_example:
        examples = gr.Examples(
            examples=[
                f'assets/example_image/{image}'
                for image in os.listdir("assets/example_image")
            ],
            inputs=[image_prompt],
            fn=preprocess_image,
            outputs=[image_prompt],
            run_on_click=True,
            examples_per_page=64,
        )

    with gr.Row(visible=False) as multiimage_example:
        examples_multi = gr.Examples(
            examples=prepare_multi_example(),
            inputs=[image_prompt],
            fn=split_image,
            outputs=[multiimage_prompt],
            run_on_click=True,
            examples_per_page=8,
        )

    # Handlers
    demo.load(start_session)
    demo.unload(end_session)
    demo.load(
        _refresh_all_asset_views,
        inputs=[],
        outputs=[asset_library_dropdown, obj_dropdown, generated_assets_status, current_asset_key, video_output, model_output, scene_asset_view, gs_object_path],
    )

    def _show_generation_mode(_input_type: str, _is_multi: bool, show_text_example: bool, show_single: bool, show_multi: bool):
        return tuple([
            _input_type,
            _is_multi,
            gr.update(visible=show_text_example),
            gr.update(visible=show_single),
            gr.update(visible=show_multi),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=True),
            gr.update(visible=False),
        ])

    def _show_scene_mode():
        return tuple([
            "scene",
            False,
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=True),
        ])

    text_input_tab.select(
        lambda: _show_generation_mode("text", False, True, False, False),
        outputs=[input_type, is_multiimage, text_example, single_image_example, multiimage_example, asset_name_group, gen_params_panel, export_panel, generate_btn, gen_output_group, scene_output_group]
    )

    single_image_input_tab.select(
        lambda: _show_generation_mode("image", False, False, True, False),
        outputs=[input_type, is_multiimage, text_example, single_image_example, multiimage_example, asset_name_group, gen_params_panel, export_panel, generate_btn, gen_output_group, scene_output_group]
    )

    multiimage_input_tab.select(
        lambda: _show_generation_mode("image", True, False, False, True),
        outputs=[input_type, is_multiimage, text_example, single_image_example, multiimage_example, asset_name_group, gen_params_panel, export_panel, generate_btn, gen_output_group, scene_output_group]
    )

    scene_editor_tab.select(
        lambda: _show_scene_mode(),
        outputs=[input_type, is_multiimage, text_example, single_image_example, multiimage_example, asset_name_group, gen_params_panel, export_panel, generate_btn, gen_output_group, scene_output_group]
    )

    scene_editor_tab.select(
        fn=_refresh_all_asset_views,
        inputs=[],
        outputs=[asset_library_dropdown, obj_dropdown, generated_assets_status, current_asset_key, video_output, model_output, scene_asset_view, gs_object_path],
    )

    obj_dropdown.change(
        _asset_key_to_scene_selection,
        inputs=[obj_dropdown],
        outputs=[current_asset_key, gs_object_path, scene_asset_view, generated_assets_status, asset_library_dropdown],
    )

    asset_library_dropdown.change(
        _select_asset_for_preview,
        inputs=[asset_library_dropdown],
        outputs=[current_asset_key, video_output, model_output, latest_object_glb, latest_object_ply, generated_assets_status, gs_object_path, scene_asset_view, obj_dropdown],
    )

    refresh_composed_preview_btn.click(
        lambda p: p or None,
        inputs=[composed_ply_view],
        outputs=[composed_ply_view],
    )

    # 绑定“合成高斯”到右侧栏输出：链接/下载/预览
    gs_compose_btn.click(
        fn=_compose_gaussian,
        inputs=[
            gs_scene_id,
            gs_scene_path,
            gs_object_path,
            gs_objects_state,
            gs_pos_x,
            gs_pos_y,
            gs_pos_z,
            gs_rot_x,
            gs_rot_y,
            gs_rot_z,
            gs_scale,
        ],
        outputs=[gs_status, gs_viewer_link_right, gs_download_right, composed_ply_view],
    )

    image_prompt.upload(
        preprocess_image,
        inputs=[image_prompt],
        outputs=[image_prompt],
    )
    multiimage_prompt.upload(
        preprocess_images,
        inputs=[multiimage_prompt],
        outputs=[multiimage_prompt],
    )

    generate_btn.click(
        get_seed,
        inputs=[randomize_seed, seed],
        outputs=[seed],
    ).then(
        generate_3d,
        inputs=[input_type, asset_name, text_prompt, image_prompt, multiimage_prompt, is_multiimage, seed, ss_guidance_strength, ss_sampling_steps, slat_guidance_strength, slat_sampling_steps, multiimage_algo],
        outputs=[output_buf, video_output, current_asset_key],
    ).then(
        lambda asset_key: (
            _asset_status_text(asset_key),
            gr.Button(interactive=True),
            gr.Button(interactive=True),
        ),
        inputs=[current_asset_key],
        outputs=[generated_assets_status, extract_glb_btn, extract_gs_btn],
    ).then(
        _refresh_all_asset_views,
        inputs=[],
        outputs=[asset_library_dropdown, obj_dropdown, generated_assets_status, current_asset_key, video_output, model_output, scene_asset_view, gs_object_path],
    )

    video_output.clear(
        lambda: tuple([gr.Button(interactive=False), gr.Button(interactive=False)]),
        outputs=[extract_glb_btn, extract_gs_btn],
    )

    refresh_latest_asset_btn.click(
        _refresh_all_asset_views,
        inputs=[],
        outputs=[asset_library_dropdown, obj_dropdown, generated_assets_status, current_asset_key, video_output, model_output, scene_asset_view, gs_object_path],
    )

    refresh_scene_assets_btn.click(
        _refresh_all_asset_views,
        inputs=[],
        outputs=[asset_library_dropdown, obj_dropdown, generated_assets_status, current_asset_key, video_output, model_output, scene_asset_view, gs_object_path],
    )

    clear_generated_assets_btn.click(
        _clear_generated_assets,
        inputs=[],
        outputs=[asset_library_dropdown, obj_dropdown, generated_assets_status, video_output, model_output, scene_asset_view, gs_object_path],
    ).then(
        lambda: (None, None, None, gr.Button(interactive=False), gr.Button(interactive=False)),
        outputs=[current_asset_key, latest_object_glb, latest_object_ply, extract_glb_btn, extract_gs_btn],
    )

    def extract_glb_and_cache(state: dict, mesh_simplify: float, texture_size: int, req: gr.Request):
        glb_path, _ = extract_glb(state, mesh_simplify, texture_size, req)
        # model_output / download_glb / latest_object_glb / scene_asset_view 都用同一个 glb 路径
        return glb_path, glb_path, glb_path, glb_path

    extract_glb_btn.click(
        extract_glb_and_cache,
        inputs=[output_buf, mesh_simplify, texture_size],
        outputs=[model_output, download_glb, latest_object_glb, scene_asset_view],
    ).then(
        lambda: gr.Button(interactive=True),
        outputs=[download_glb],
    )

    def extract_gaussian_and_cache(state: dict, req: gr.Request):
        ply_path, _ = extract_gaussian(state, req)
        # model_output / download_gs / latest_object_ply / scene_asset_view / gs_object_path 都用同一个 ply 路径
        # 同时刷新“物体下拉框”的选中值（避免你手动点上传）
        
        # 为“导出高斯”按钮也生成预览链接（如果需要的话，目前只是返回路径）
        # viewer_url = _viewer_url_for_ply(str(req.session_hash), ply_path, req)
        
        try:
            choices = []
            if "obj_dropdown" in globals():
                # 不依赖 globals 里有组件；这里只占位，真正更新靠前端刷新按钮
                choices = []
        except Exception:
            pass
        return ply_path, ply_path, ply_path, ply_path, ply_path

    extract_gs_btn.click(
        extract_gaussian_and_cache,
        inputs=[output_buf],
        outputs=[model_output, download_gs, latest_object_ply, scene_asset_view, gs_object_path],
    ).then(
        lambda: gr.Button(interactive=True),
        outputs=[download_gs],
    )

    model_output.clear(
        lambda: gr.Button(interactive=False),
        outputs=[download_glb],
    )

    # Add footer
    gr.HTML("""
    <div style="text-align: center; margin-top: 40px; padding: 20px; background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); border-radius: 15px;">
        <p style="margin: 0; color: #666; font-size: 0.9em;">
            🎨 基于 <a href="https://trellis3d.github.io/" target="_blank" style="color: #667eea; text-decoration: none; font-weight: 500;">TRELLIS</a> 构建
            | 🏗️ 由 AI 驱动的 3D 生成技术
            | ⚡ 支持多种输入格式
        </p>
        <div style="margin-top: 10px;">
            <span style="background: #667eea; color: white; padding: 5px 10px; border-radius: 12px; font-size: 0.8em;">Microsoft Research</span>
            <span style="background: #764ba2; color: white; padding: 5px 10px; border-radius: 12px; font-size: 0.8em; margin-left: 10px;">CVPR 2025</span>
        </div>
    </div>
    """)


# Launch the Gradio app
if __name__ == "__main__":
    # --- 加载本地翻译模型 ---
    print("正在从本地加载中英文翻译模型...")
    model_path = "./models/translation_zh_en"
    try:
        # 显式指定本地路径
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        translator = pipeline("translation_zh_to_en", model=model, tokenizer=tokenizer, device="cpu")
        print("翻译模型加载成功。")
    except Exception as e:
        print(f"翻译模型加载失败: {e}。请确保模型文件已放在 {model_path}")
        translator = None

    # Initialize both pipelines (using local models)
    print("\n" + "="*60)
    print("📦 正在加载图片生成模型 (TRELLIS-image-large)...")
    print("   这可能需要 1~3 分钟，请耐心等待...")
    print("="*60)
    image_pipeline = TrellisImageTo3DPipeline.from_pretrained("./models/TRELLIS-image-large")
    print("✅ 图片模型加载完成，正在搬到 GPU...")
    image_pipeline.cuda()
    print("✅ 图片模型已就绪\n")

    print("="*60)
    print("📦 正在加载文字生成模型 (TRELLIS-text-large)...")
    print("   这可能需要 1~3 分钟，请耐心等待...")
    print("="*60)
    text_pipeline = TrellisTextTo3DPipeline.from_pretrained("./models/TRELLIS-text-large")
    print("✅ 文字模型加载完成，正在搬到 GPU...")
    text_pipeline.cuda()
    print("✅ 文字模型已就绪\n")

    # 预启动交互编辑器静态服务，避免首次点击编辑器时才懒加载导致卡顿/404。
    _ensure_viewer_server()

    port = int(os.environ.get("GRADIO_SERVER_PORT", os.environ.get("PORT", "7860")))
    host = os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1")
    print(f"TRELLIS unified app is ready. Open: http://127.0.0.1:{port}")
    print(f"GS viewer is ready. Open: http://127.0.0.1:{_VIEWER_PORT_ACTUAL or _VIEWER_PORT}/viewer/index.html")
    demo.queue(max_size=32, default_concurrency_limit=2)
    demo.launch(server_name=host, server_port=port, share=False)
