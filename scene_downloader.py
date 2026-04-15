import os
import json
import requests
import zipfile
from pathlib import Path
from typing import Dict, List, Optional
import shutil

class SceneDownloader:
    """3D场景下载器，用于下载预设场景"""

    def __init__(self, scenes_dir: str = "./scenes"):
        self.scenes_dir = Path(scenes_dir)
        self.scenes_dir.mkdir(exist_ok=True)

        # 预定义的场景库
        self.available_scenes = {
            "classroom": {
                "name": "现代化教室",
                "description": "一个现代化的教室场景，包含课桌、黑板和窗户",
                "download_url": "https://example.com/scenes/classroom.zip",  # 需要替换为实际URL
                "preview_image": "classroom_preview.jpg",
                "tags": ["教育", "室内", "现代"]
            },
            "space_battlefield": {
                "name": "太空战场",
                "description": "未来科幻太空战斗场景，包含飞船和太空站",
                "download_url": "https://example.com/scenes/space_battlefield.zip",
                "preview_image": "space_preview.jpg",
                "tags": ["科幻", "太空", "战斗"]
            },
            "cozy_bedroom": {
                "name": "温馨卧室",
                "description": "舒适的卧室场景，包含床、书桌和装饰品",
                "download_url": "https://example.com/scenes/cozy_bedroom.zip",
                "preview_image": "bedroom_preview.jpg",
                "tags": ["家居", "温馨", "放松"]
            },
            "forest_clearing": {
                "name": "森林空地",
                "description": "宁静的森林空地，周围环绕着树木和阳光",
                "download_url": "https://example.com/scenes/forest_clearing.zip",
                "preview_image": "forest_preview.jpg",
                "tags": ["自然", "户外", "宁静"]
            },
            "minimalist_office": {
                "name": "极简办公室",
                "description": "现代极简风格的办公室，简洁而高效",
                "download_url": "https://example.com/scenes/minimalist_office.zip",
                "preview_image": "office_preview.jpg",
                "tags": ["办公", "现代", "极简"]
            },
            "beach_sunset": {
                "name": "海滩日落",
                "description": "美丽的海滩日落场景，沙滩、海浪和落日",
                "download_url": "https://example.com/scenes/beach_sunset.zip",
                "preview_image": "beach_preview.jpg",
                "tags": ["自然", "海滩", "日落"]
            },
            "cyberpunk_city": {
                "name": "赛博朋克城市",
                "description": "未来感十足的赛博朋克城市街景",
                "download_url": "https://example.com/scenes/cyberpunk_city.zip",
                "preview_image": "cyberpunk_preview.jpg",
                "tags": ["科幻", "城市", "未来"]
            },
            "medieval_castle": {
                "name": "中世纪城堡",
                "description": "古老的中世纪城堡，充满历史韵味",
                "download_url": "https://example.com/scenes/medieval_castle.zip",
                "preview_image": "castle_preview.jpg",
                "tags": ["历史", "城堡", "中世纪"]
            }
        }

    def download_scene(self, scene_id: str, progress_callback=None) -> bool:
        """下载指定的场景"""
        if scene_id not in self.available_scenes:
            print(f"场景 {scene_id} 不存在")
            return False

    def import_local_scene(
        self,
        scene_id: str,
        scene_name: str,
        description: str,
        tags: Optional[str],
        scene_file: object,
    ) -> bool:
        """
        导入本地场景文件到 ./scenes/<scene_id>/scene.<ext>，并生成 scene_info.json。

        - 支持: .glb / .gltf / .zip（zip内包含glb/gltf）
        - scene_file 兼容 gr.File 返回的 str / dict / 临时文件对象
        """
        scene_id = (scene_id or "").strip()
        if not scene_id:
            raise ValueError("scene_id 不能为空")

        scene_dir = self.scenes_dir / scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)

        # 解析 gradio File 返回值
        local_path: Optional[str] = None
        if isinstance(scene_file, str):
            local_path = scene_file
        elif isinstance(scene_file, dict):
            local_path = scene_file.get("path") or scene_file.get("name")
        else:
            local_path = getattr(scene_file, "name", None) or getattr(scene_file, "path", None)

        if not local_path:
            raise ValueError("无法解析场景文件路径")

        src = Path(local_path)
        if not src.exists():
            raise ValueError(f"场景文件不存在: {src}")

        chosen_scene_file: Optional[Path] = None
        if src.suffix.lower() == ".zip":
            extract_dir = scene_dir / "_import_tmp"
            if extract_dir.exists():
                shutil.rmtree(extract_dir)
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(src, "r") as zf:
                zf.extractall(extract_dir)

            candidates = list(extract_dir.rglob("*.glb")) + list(extract_dir.rglob("*.gltf"))
            if not candidates:
                raise ValueError("zip 内未找到 .glb/.gltf 场景文件")
            chosen_scene_file = candidates[0]
        elif src.suffix.lower() in [".glb", ".gltf"]:
            chosen_scene_file = src
        else:
            raise ValueError("仅支持 .glb / .gltf / .zip")

        dst_ext = chosen_scene_file.suffix.lower()
        dst_scene_path = scene_dir / f"scene{dst_ext}"
        shutil.copy2(chosen_scene_file, dst_scene_path)

        # 清理临时目录
        tmp_dir = scene_dir / "_import_tmp"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)

        tag_list = []
        if tags:
            tag_list = [t.strip() for t in str(tags).split(",") if t.strip()]

        scene_config = {
            "id": scene_id,
            "name": scene_name or scene_id,
            "description": description or "",
            "tags": tag_list,
            "scene_file": str(dst_scene_path),
            "dimensions": {"width": 20, "height": 10, "depth": 20},
            "placement_zones": self._get_default_zones(scene_id),
            "objects": [],
            "created_by": "TRELLIS Scene Importer",
            "version": "1.0",
        }

        config_file = scene_dir / "scene_info.json"
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(scene_config, f, indent=2, ensure_ascii=False)

        return True

        scene_info = self.available_scenes[scene_id]
        scene_dir = self.scenes_dir / scene_id

        try:
            print(f"开始下载场景: {scene_info['name']}")

            # 这里应该是实际的下载逻辑
            # 由于没有真实的下载URL，我们创建一个示例场景
            self._create_sample_scene(scene_id, scene_info)

            print(f"场景 {scene_info['name']} 下载完成")
            return True

        except Exception as e:
            print(f"下载场景失败: {e}")
            return False

    def _create_sample_scene(self, scene_id: str, scene_info: Dict):
        """创建示例场景（用于演示）"""
        scene_dir = self.scenes_dir / scene_id
        scene_dir.mkdir(exist_ok=True)

        # 创建场景信息文件
        scene_config = {
            "id": scene_id,
            "name": scene_info["name"],
            "description": scene_info["description"],
            "tags": scene_info["tags"],
            "dimensions": {"width": 20, "height": 10, "depth": 20},
            "placement_zones": self._get_default_zones(scene_id),
            "objects": [],
            "created_by": "TRELLIS Scene Downloader",
            "version": "1.0"
        }

        # 保存场景配置
        config_file = scene_dir / "scene_info.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(scene_config, f, indent=2, ensure_ascii=False)

        # 创建一个简单的场景描述文件（占位符）
        readme_file = scene_dir / "README.md"
        with open(readme_file, 'w', encoding='utf-8') as f:
            f.write(f"# {scene_info['name']}\n\n")
            f.write(f"{scene_info['description']}\n\n")
            f.write("## 放置区域\n\n")
            for zone in scene_config["placement_zones"]:
                f.write(f"- {zone['name']}: {zone['description']}\n")

        print(f"创建示例场景: {scene_id}")

    def _get_default_zones(self, scene_id: str) -> List[Dict]:
        """获取场景的默认放置区域"""
        # 根据场景类型定义不同的放置区域
        if "classroom" in scene_id:
            return [
                {"name": "teacher_desk", "position": [0, -8, 0.5], "size": [2, 1, 1], "description": "讲台区域"},
                {"name": "student_desks", "position": [0, 2, 0.3], "size": [8, 6, 0.6], "description": "学生桌区域"},
                {"name": "blackboard_area", "position": [0, -9.5, 2], "size": [6, 0.5, 2], "description": "黑板区域"}
            ]
        elif "bedroom" in scene_id:
            return [
                {"name": "bed", "position": [0, 0, 0.2], "size": [2, 2, 0.4], "description": "床边区域"},
                {"name": "desk", "position": [3, -2, 0.5], "size": [1.5, 1, 1], "description": "书桌区域"},
                {"name": "floor_space", "position": [0, 2, 0], "size": [6, 4, 0.1], "description": "地板空间"}
            ]
        elif "office" in scene_id:
            return [
                {"name": "desk", "position": [0, 0, 0.5], "size": [2, 1.5, 1], "description": "办公桌区域"},
                {"name": "meeting_area", "position": [0, 3, 0.2], "size": [4, 4, 0.4], "description": "会议区域"},
                {"name": "shelves", "position": [-4, 0, 1.5], "size": [1, 6, 3], "description": "书架区域"}
            ]
        else:
            # 默认放置区域
            return [
                {"name": "center", "position": [0, 0, 0], "size": [8, 8, 4], "description": "中央区域"},
                {"name": "front_left", "position": [-3, 3, 0], "size": [2, 2, 2], "description": "左前区域"},
                {"name": "front_right", "position": [3, 3, 0], "size": [2, 2, 2], "description": "右前区域"},
                {"name": "back_left", "position": [-3, -3, 0], "size": [2, 2, 2], "description": "左后区域"},
                {"name": "back_right", "position": [3, -3, 0], "size": [2, 2, 2], "description": "右后区域"}
            ]

    def list_available_scenes(self) -> Dict[str, Dict]:
        """列出所有可用的场景"""
        return self.available_scenes

    def get_scene_info(self, scene_id: str) -> Optional[Dict]:
        """获取场景信息"""
        scene_dir = self.scenes_dir / scene_id
        info_file = scene_dir / "scene_info.json"

        if info_file.exists():
            with open(info_file, 'r', encoding='utf-8') as f:
                return json.load(f)

        return None

    def list_downloaded_scenes(self) -> List[str]:
        """列出已下载的场景"""
        scenes = []
        for item in self.scenes_dir.iterdir():
            if item.is_dir() and (item / "scene_info.json").exists():
                scenes.append(item.name)
        return scenes
