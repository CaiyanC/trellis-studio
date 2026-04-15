import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import trimesh


class ObjectPlacer:
    """3D物体放置工具：将生成的物体（glb）记录并复制到某个场景目录下。"""

    def __init__(self, scenes_dir: str = "./scenes", output_dir: str = "./placed_objects"):
        self.scenes_dir = Path(scenes_dir)
        self.output_dir = Path(output_dir)
        self.scenes_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)

    def load_scene_info(self, scene_id: str) -> Optional[Dict]:
        """加载场景信息；不存在则返回默认模板（不创建目录）。"""
        scene_dir = self.scenes_dir / scene_id
        info_file = scene_dir / "scene_info.json"

        if info_file.exists():
            with open(info_file, "r", encoding="utf-8") as f:
                return json.load(f)

        return {
            "id": scene_id,
            "name": scene_id.replace("_", " ").title(),
            "dimensions": {"width": 10, "height": 10, "depth": 10},
            "scene_file": None,
            "placement_zones": [
                {"name": "center", "position": [0, 0, 0], "size": [8, 8, 2]},
                {"name": "corner_front_left", "position": [-3, 3, 0], "size": [2, 2, 2]},
                {"name": "corner_front_right", "position": [3, 3, 0], "size": [2, 2, 2]},
                {"name": "corner_back_left", "position": [-3, -3, 0], "size": [2, 2, 2]},
                {"name": "corner_back_right", "position": [3, -3, 0], "size": [2, 2, 2]},
            ],
            "objects": [],
        }

    def place_object(
        self,
        scene_id: str,
        object_path: str,
        position: Optional[List[float]] = None,
        rotation: Optional[List[float]] = None,
        scale: float = 1.0,
        placement_zone: str = "center",
    ) -> bool:
        """将物体放置到场景中（记录+复制文件），不做网格合并。"""
        try:
            scene_info = self.load_scene_info(scene_id)
            if not scene_info:
                print(f"无法加载场景 {scene_id} 的信息")
                return False

            if position is None:
                position = self._find_best_position(scene_info, placement_zone)
            if rotation is None:
                rotation = [0, 0, 0]

            object_path_p = Path(object_path)
            if not object_path_p.exists():
                print(f"物体文件不存在: {object_path}")
                return False

            scene_dir = self.scenes_dir / scene_id
            scene_dir.mkdir(exist_ok=True)

            placed_object_path = scene_dir / f"placed_{object_path_p.name}"
            shutil.copy2(object_path_p, placed_object_path)

            object_info = {
                "id": f"object_{len(scene_info.get('objects', []))}",
                "original_path": str(object_path_p),
                "placed_path": str(placed_object_path),
                "position": position,
                "rotation": rotation,
                "scale": scale,
                "placement_zone": placement_zone,
                "timestamp": str(object_path_p.stat().st_mtime),
            }

            scene_info.setdefault("objects", []).append(object_info)

            info_file = scene_dir / "scene_info.json"
            with open(info_file, "w", encoding="utf-8") as f:
                json.dump(scene_info, f, indent=2, ensure_ascii=False)

            print(f"物体已成功放置到场景 {scene_id} 中：{placed_object_path.name}")
            return True
        except Exception as e:
            print(f"放置物体时出错: {e}")
            return False

    def _find_best_position(self, scene_info: Dict, placement_zone: str) -> List[float]:
        zones = scene_info.get("placement_zones", [])
        target_zone = next((z for z in zones if z.get("name") == placement_zone), None)
        if not target_zone:
            return [0, 0, 0]

        zone_pos = target_zone["position"]
        zone_size = target_zone["size"]

        x = zone_pos[0] + np.random.uniform(-zone_size[0] / 2, zone_size[0] / 2)
        y = zone_pos[1] + np.random.uniform(-zone_size[1] / 2, zone_size[1] / 2)
        z = zone_pos[2] + np.random.uniform(0, zone_size[2] / 2)
        return [round(float(x), 2), round(float(y), 2), round(float(z), 2)]

    def list_placed_objects(self, scene_id: str) -> List[Dict]:
        """列出场景中已放置的物体"""
        scene_info = self.load_scene_info(scene_id)
        if scene_info:
            return scene_info.get("objects", [])
        return []

    def remove_object(self, scene_id: str, object_id: str) -> bool:
        """从场景中移除物体（同时删除已复制文件）"""
        try:
            scene_dir = self.scenes_dir / scene_id
            info_file = scene_dir / "scene_info.json"
            if not info_file.exists():
                return False

            with open(info_file, "r", encoding="utf-8") as f:
                scene_info = json.load(f)

            objects = scene_info.get("objects", [])
            for i, obj in enumerate(objects):
                if obj.get("id") == object_id:
                    placed_path = obj.get("placed_path")
                    if placed_path and Path(placed_path).exists():
                        Path(placed_path).unlink()
                    objects.pop(i)
                    break
            else:
                return False

            with open(info_file, "w", encoding="utf-8") as f:
                json.dump(scene_info, f, indent=2, ensure_ascii=False)
            return True
        except Exception:
            return False

    def list_downloaded_scenes(self) -> List[str]:
        """列出已导入/下载的场景（存在scene_info.json的目录）"""
        scenes: List[str] = []
        if not self.scenes_dir.exists():
            return scenes
        for item in self.scenes_dir.iterdir():
            if item.is_dir() and (item / "scene_info.json").exists():
                scenes.append(item.name)
        scenes.sort()
        return scenes

    def compose_scene_glb(self, scene_id: str, output_path: Optional[str] = None) -> str:
        """
        将场景底座(scene.glb/gltf) + 已放置物体(placed_*.glb) 按 scene_info.json 的位姿合成为一个新的GLB文件。

        注意：这是“导出预览/导出合成”的能力，仍然不等同于物理仿真（碰撞体/刚体等需要在引擎侧处理）。
        """
        scene_dir = self.scenes_dir / scene_id
        info_file = scene_dir / "scene_info.json"
        if not info_file.exists():
            raise FileNotFoundError(f"scene_info.json 不存在: {info_file}")

        with open(info_file, "r", encoding="utf-8") as f:
            scene_info = json.load(f)

        # 解析场景底座文件
        scene_file = scene_info.get("scene_file")
        base_scene_path: Optional[Path] = None
        if scene_file:
            base_scene_path = Path(scene_file)
        else:
            # 兼容：默认找 ./scenes/<scene_id>/scene.glb 或 scene.gltf
            for cand in [scene_dir / "scene.glb", scene_dir / "scene.gltf"]:
                if cand.exists():
                    base_scene_path = cand
                    break

        if not base_scene_path or not base_scene_path.exists():
            raise FileNotFoundError(f"找不到场景底座文件（scene.glb/scene.gltf）: {scene_id}")

        # 输出路径
        if output_path is None:
            output_path = str(scene_dir / "composed_scene.glb")
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        # 合成
        composed = trimesh.Scene()

        base = trimesh.load(base_scene_path, force="scene")
        if isinstance(base, trimesh.Scene):
            composed = trimesh.util.concatenate([base, composed]) if composed.geometry else base
        else:
            composed.add_geometry(base)

        objects = scene_info.get("objects", [])
        for obj in objects:
            placed_path = obj.get("placed_path") or obj.get("original_path")
            if not placed_path:
                continue
            p = Path(placed_path)
            if not p.exists():
                continue

            geom = trimesh.load(p, force="scene")

            pos = obj.get("position") or [0, 0, 0]
            rot = obj.get("rotation") or [0, 0, 0]  # degrees
            scl = float(obj.get("scale") or 1.0)

            # 4x4 transform: T * Rz * Ry * Rx * S
            T = trimesh.transformations.translation_matrix(pos)
            Rx = trimesh.transformations.rotation_matrix(np.deg2rad(rot[0]), [1, 0, 0])
            Ry = trimesh.transformations.rotation_matrix(np.deg2rad(rot[1]), [0, 1, 0])
            Rz = trimesh.transformations.rotation_matrix(np.deg2rad(rot[2]), [0, 0, 1])
            S = np.eye(4, dtype=float)
            S[0, 0] = scl
            S[1, 1] = scl
            S[2, 2] = scl
            M = T @ (Rz @ (Ry @ (Rx @ S)))

            if isinstance(geom, trimesh.Scene):
                # 将scene整体应用变换：对每个geometry的node transform叠乘
                for node_name in geom.graph.nodes_geometry:
                    old = geom.graph.get(node_name)[0]
                    geom.graph.update(node_name, matrix=M @ old)
                composed = trimesh.util.concatenate([composed, geom])
            else:
                geom.apply_transform(M)
                composed.add_geometry(geom)

        composed.export(out_p)
        return str(out_p)

    def export_scene_with_objects(self, scene_id: str, output_format: str = "gltf") -> Optional[str]:
        """导出包含物体的完整场景"""
        try:
            scene_info = self.load_scene_info(scene_id)
            if not scene_info:
                return None

            scene_dir = self.scenes_dir / scene_id
            output_scene_dir = self.output_dir / f"{scene_id}_with_objects"
            output_scene_dir.mkdir(exist_ok=True)

            # 这里应该实现实际的场景合成逻辑
            # 由于需要复杂的3D处理，这里提供一个简化的实现

            # 创建场景描述文件
            scene_description = {
                "scene_info": scene_info,
                "export_format": output_format,
                "export_time": str(Path().cwd().stat().st_mtime),
                "instructions": """
要创建完整的场景，请使用以下3D软件之一：

1. Blender (免费): https://www.blender.org/
   - 导入场景文件
   - 导入每个物体的GLB文件
   - 根据scene_info.json中的位置信息放置物体
   - 导出为最终场景

2. Three.js (Web): https://threejs.org/
   - 使用GLTFLoader加载场景和物体
   - 根据位置信息设置物体变换
   - 渲染完整场景

3. Unity (游戏引擎): https://unity.com/
   - 导入GLTF文件
   - 使用C#脚本放置物体
   - 构建场景

物体位置信息存储在 scene_info.json 的 objects 数组中。
每个物体都有 position, rotation, scale 属性。
                """
            }

            output_file = output_scene_dir / "scene_description.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(scene_description, f, indent=2, ensure_ascii=False)

            # 复制所有物体文件
            for obj in scene_info["objects"]:
                if "placed_path" in obj and Path(obj["placed_path"]).exists():
                    shutil.copy2(obj["placed_path"], output_scene_dir)

            print(f"场景导出完成: {output_scene_dir}")
            return str(output_scene_dir)

        except Exception as e:
            print(f"导出场景时出错: {e}")
            return None

    def list_placed_objects(self, scene_id: str) -> List[Dict]:
        """列出场景中已放置的物体"""
        scene_info = self.load_scene_info(scene_id)
        if scene_info:
            return scene_info.get("objects", [])
        return []

    def list_downloaded_scenes(self) -> List[str]:
        """列出已下载的场景"""
        scenes = []
        for item in self.scenes_dir.iterdir():
            if item.is_dir() and (item / "scene_info.json").exists():
                scenes.append(item.name)
        return scenes

    def remove_object(self, scene_id: str, object_id: str) -> bool:
        """从场景中移除物体"""
        try:
            scene_info = self.load_scene_info(scene_id)
            if not scene_info:
                return False

            # 找到并移除物体
            objects = scene_info.get("objects", [])
            for i, obj in enumerate(objects):
                if obj["id"] == object_id:
                    # 删除文件
                    if "placed_path" in obj and Path(obj["placed_path"]).exists():
                        Path(obj["placed_path"]).unlink()

                    # 从列表中移除
                    objects.pop(i)

                    # 保存更新后的场景信息
                    scene_dir = self.scenes_dir / scene_id
                    info_file = scene_dir / "scene_info.json"
                    with open(info_file, 'w', encoding='utf-8') as f:
                        json.dump(scene_info, f, indent=2, ensure_ascii=False)

                    print(f"物体 {object_id} 已从场景 {scene_id} 中移除")
                    return True

            print(f"在场景 {scene_id} 中找不到物体 {object_id}")
            return False

        except Exception as e:
            print(f"移除物体时出错: {e}")
            return False

if __name__ == "__main__":
    # 测试物体放置器
    placer = ObjectPlacer()

    # 模拟放置一个物体
    test_scene = "test_scene"
    test_object = "./sample.glb"  # 假设有这个文件

    if Path(test_object).exists():
        success = placer.place_object(
            scene_id=test_scene,
            object_path=test_object,
            placement_zone="center",
            scale=0.8
        )

        if success:
            print("物体放置测试成功")

            # 列出已放置的物体
            objects = placer.list_placed_objects(test_scene)
            print(f"场景中的物体: {len(objects)} 个")

            # 导出场景
            export_path = placer.export_scene_with_objects(test_scene)
            if export_path:
                print(f"场景已导出到: {export_path}")
    else:
        print("测试用的GLB文件不存在，请先生成一个3D物体")
