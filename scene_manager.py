import os
import requests
import zipfile
import shutil
from pathlib import Path
import json
from typing import List, Dict, Optional

class SceneManager:
    """3D场景管理器，用于下载和管理3D场景资源"""

    def __init__(self, scenes_dir: str = "./scenes"):
        self.scenes_dir = Path(scenes_dir)
        self.scenes_dir.mkdir(exist_ok=True)
        self.scenes_info_file = self.scenes_dir / "scenes_info.json"
        self._load_scenes_info()

    def _load_scenes_info(self):
        """加载场景信息"""
        if self.scenes_info_file.exists():
            with open(self.scenes_info_file, 'r', encoding='utf-8') as f:
                self.scenes_info = json.load(f)
        else:
            self.scenes_info = {}
            self._create_default_scenes()

    def _save_scenes_info(self):
        """保存场景信息"""
        with open(self.scenes_info_file, 'w', encoding='utf-8') as f:
            json.dump(self.scenes_info, f, indent=2, ensure_ascii=False)

    def _create_default_scenes(self):
        """创建默认场景列表"""
        self.scenes_info = {
            "classroom": {
                "name": "现代化教室",
                "description": "温馨的现代化教室场景，适合放置学习用品",
                "category": "教育",
                "download_url": "https://example.com/classroom.zip",
                "file_format": "gltf",
                "tags": ["教育", "室内", "学习"],
                "downloaded": False
            },
            "space_station": {
                "name": "太空空间站",
                "description": "未来感十足的太空空间站，适合放置科幻元素",
                "category": "科幻",
                "download_url": "https://example.com/space_station.zip",
                "file_format": "gltf",
                "tags": ["科幻", "太空", "未来"],
                "downloaded": False
            },
            "cozy_library": {
                "name": "温馨图书馆",
                "description": "舒适的图书馆环境，适合放置书籍和阅读用品",
                "category": "文化",
                "download_url": "https://example.com/library.zip",
                "file_format": "gltf",
                "tags": ["文化", "室内", "阅读"],
                "downloaded": False
            },
            "garden_therapy": {
                "name": "花园疗愈空间",
                "description": "宁静的花园场景，适合放置植物和疗愈元素",
                "category": "疗愈",
                "download_url": "https://example.com/garden.zip",
                "file_format": "gltf",
                "tags": ["疗愈", "自然", "花园"],
                "downloaded": False
            },
            "medieval_castle": {
                "name": "中世纪城堡",
                "description": "古老的中世纪城堡，适合放置古典元素",
                "category": "历史",
                "download_url": "https://example.com/castle.zip",
                "file_format": "gltf",
                "tags": ["历史", "城堡", "中世纪"],
                "downloaded": False
            }
        }
        self._save_scenes_info()

    def list_available_scenes(self, category: Optional[str] = None) -> List[Dict]:
        """列出可用的场景"""
        scenes = []
        for scene_id, info in self.scenes_info.items():
            if category is None or info.get("category") == category:
                scenes.append({
                    "id": scene_id,
                    **info
                })
        return scenes

    def get_categories(self) -> List[str]:
        """获取所有分类"""
        return list(set(info["category"] for info in self.scenes_info.values()))

    def download_scene(self, scene_id: str, progress_callback=None) -> bool:
        """下载场景"""
        if scene_id not in self.scenes_info:
            print(f"场景 {scene_id} 不存在")
            return False

        scene_info = self.scenes_info[scene_id]
        if scene_info["downloaded"]:
            print(f"场景 {scene_id} 已经下载过了")
            return True

        # 创建场景目录
        scene_dir = self.scenes_dir / scene_id
        scene_dir.mkdir(exist_ok=True)

        try:
            print(f"正在下载场景: {scene_info['name']}")
            # 这里应该实现实际的下载逻辑
            # 由于示例URL不可用，我们创建一个占位符

            # 创建一个简单的占位符场景文件
            placeholder_content = f"""
# 这是一个占位符场景文件
# 场景ID: {scene_id}
# 场景名称: {scene_info['name']}
# 描述: {scene_info['description']}
# 实际下载请访问: https://sketchfab.com/ 或 https://poly.google.com/
# 推荐免费3D场景资源网站:
# 1. Sketchfab (https://sketchfab.com/)
# 2. Google Poly (https://poly.google.com/)
# 3. Turbosquid (https://www.turbosquid.com/)
# 4. CGTrader (https://www.cgtrader.com/)
# 5. Free3D (https://free3d.com/)
"""

            with open(scene_dir / "README.txt", 'w', encoding='utf-8') as f:
                f.write(placeholder_content)

            # 标记为已下载
            scene_info["downloaded"] = True
            scene_info["local_path"] = str(scene_dir)
            self._save_scenes_info()

            print(f"场景 {scene_id} 下载完成")
            return True

        except Exception as e:
            print(f"下载场景 {scene_id} 时出错: {e}")
            return False

    def get_scene_path(self, scene_id: str) -> Optional[str]:
        """获取场景的本地路径"""
        if scene_id in self.scenes_info and self.scenes_info[scene_id]["downloaded"]:
            return self.scenes_info[scene_id].get("local_path")
        return None

    def add_custom_scene(self, scene_id: str, name: str, description: str,
                        category: str, local_path: str, tags: List[str] = None):
        """添加自定义场景"""
        self.scenes_info[scene_id] = {
            "name": name,
            "description": description,
            "category": category,
            "download_url": "",
            "file_format": "gltf",
            "tags": tags or [],
            "downloaded": True,
            "local_path": local_path
        }
        self._save_scenes_info()
        print(f"自定义场景 {scene_id} 已添加")

# 推荐的免费3D场景资源网站
SCENE_RESOURCES = {
    "Sketchfab": {
        "url": "https://sketchfab.com/",
        "description": "最大的3D模型分享平台，有大量免费场景",
        "categories": ["所有类型", "教育", "科幻", "自然", "建筑"]
    },
    "Google Poly": {
        "url": "https://poly.google.com/",
        "description": "Google的3D资源库，适合VR/AR应用",
        "categories": ["室内", "户外", "教育"]
    },
    "Turbosquid": {
        "url": "https://www.turbosquid.com/",
        "description": "专业3D模型市场，有免费资源",
        "categories": ["建筑", "室内", "工业"]
    },
    "CGTrader": {
        "url": "https://www.cgtrader.com/",
        "description": "3D模型交易平台，包含免费模型",
        "categories": ["所有类型"]
    },
    "Free3D": {
        "url": "https://free3d.com/",
        "description": "免费3D模型下载网站",
        "categories": ["游戏", "建筑", "室内"]
    },
    "OpenGameArt": {
        "url": "https://opengameart.org/",
        "description": "开源游戏美术资源",
        "categories": ["游戏", "科幻", "奇幻"]
    }
}

if __name__ == "__main__":
    # 测试场景管理器
    manager = SceneManager()

    print("可用的场景分类:")
    for category in manager.get_categories():
        print(f"  - {category}")

    print("\n所有可用场景:")
    scenes = manager.list_available_scenes()
    for scene in scenes:
        print(f"  {scene['id']}: {scene['name']} ({scene['category']})")
        print(f"    {scene['description']}")

    print("\n推荐的3D场景资源网站:")
    for name, info in SCENE_RESOURCES.items():
        print(f"  {name}: {info['url']}")
        print(f"    {info['description']}")
        print(f"    分类: {', '.join(info['categories'])}")
