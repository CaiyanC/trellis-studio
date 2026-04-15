import gradio as gr
import json
from pathlib import Path
from typing import List, Dict, Optional
import shutil

from scene_downloader import SceneDownloader
from object_placer import ObjectPlacer

class SceneEditor:
    """3D场景编辑器"""

    def __init__(self):
        self.scene_downloader = SceneDownloader()
        self.object_placer = ObjectPlacer()

    def create_scene_editor_interface(self):
        """创建场景编辑器Gradio界面"""

        with gr.Blocks(title="TRELLIS 场景编辑器", theme=gr.themes.Soft()) as scene_editor:
            gr.HTML("""
            <div style="text-align: center; margin-bottom: 30px;">
                <h1 style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.5em; font-weight: 700; margin: 0;">
                    🏗️ TRELLIS 场景编辑器
                </h1>
                <p style="font-size: 1.1em; color: #666; margin: 10px 0;">
                    将生成的3D物体放置到预设场景中，创造独特的3D世界
                </p>
            </div>
            """)

            with gr.Row():
                # 左侧：场景管理
                with gr.Column(scale=1):
                    gr.HTML("<h3 style='color: #667eea; margin-bottom: 15px;'>📁 场景管理</h3>")

                    # 场景下载区域
                    with gr.Accordion("🌐 下载预设场景", open=False):
                        available_scenes = self.scene_downloader.list_available_scenes()

                        scene_list = gr.Radio(
                            choices=[f"{sid}: {info['name']} - {info['description']}"
                                    for sid, info in available_scenes.items()],
                            label="可用场景",
                            info="选择要下载的场景"
                        )

                        download_btn = gr.Button("⬇️ 下载场景", variant="secondary")
                        download_status = gr.Textbox(label="下载状态", interactive=False)

                    # 已下载场景选择
                    downloaded_scenes = self.object_placer.list_downloaded_scenes()
                    selected_scene = gr.Dropdown(
                        choices=downloaded_scenes,
                        label="选择场景",
                        info="选择已下载的场景进行编辑"
                    )

                    scene_info_display = gr.JSON(label="场景信息")

                    # 刷新场景列表
                    refresh_btn = gr.Button("🔄 刷新场景列表", variant="secondary")

                # 中间：物体放置
                with gr.Column(scale=1):
                    gr.HTML("<h3 style='color: #764ba2; margin-bottom: 15px;'>🎯 物体放置</h3>")

                    # 物体选择
                    generated_objects = self._find_generated_objects()
                    selected_object = gr.Dropdown(
                        choices=generated_objects,
                        label="选择要放置的物体",
                        info="选择已生成的3D物体文件"
                    )

                    # 放置参数
                    with gr.Group():
                        placement_zone = gr.Dropdown(
                            choices=[],  # 将由场景信息动态填充
                            label="放置区域",
                            info="选择场景中的放置区域"
                        )

                        with gr.Row():
                            pos_x = gr.Slider(-10, 10, value=0, step=0.1, label="X位置")
                            pos_y = gr.Slider(-10, 10, value=0, step=0.1, label="Y位置")
                            pos_z = gr.Slider(-5, 10, value=0, step=0.1, label="Z位置")

                        with gr.Row():
                            rot_x = gr.Slider(-180, 180, value=0, step=5, label="X旋转")
                            rot_y = gr.Slider(-180, 180, value=0, step=5, label="Y旋转")
                            rot_z = gr.Slider(-180, 180, value=0, step=5, label="Z旋转")

                        scale = gr.Slider(0.1, 5.0, value=1.0, step=0.1, label="缩放")

                        auto_place = gr.Checkbox(value=True, label="自动选择最佳位置")

                    place_btn = gr.Button("🎯 放置物体", variant="primary", size="lg")
                    place_status = gr.Textbox(label="放置状态", interactive=False)

                # 右侧：场景预览和导出
                with gr.Column(scale=1):
                    gr.HTML("<h3 style='color: #28a745; margin-bottom: 15px;'>🎬 场景预览</h3>")

                    # 场景中的物体列表
                    scene_objects = gr.Dataframe(
                        headers=["物体", "位置", "旋转", "缩放", "区域"],
                        label="场景中的物体",
                        interactive=False
                    )

                    # 导出选项
                    with gr.Accordion("💾 导出场景", open=False):
                        export_format = gr.Radio(
                            choices=["json", "html", "blend"],
                            value="json",
                            label="导出格式"
                        )

                        export_btn = gr.Button("📤 导出场景", variant="primary")
                        export_status = gr.Textbox(label="导出状态", interactive=False)

                    # 场景统计
                    scene_stats = gr.JSON(label="场景统计")

            # 事件处理
            def update_scene_info(scene_id):
                if not scene_id:
                    return None, [], {}

                scene_info = self.scene_downloader.get_scene_info(scene_id)
                if not scene_info:
                    return None, [], {}

                # 获取放置区域
                zones = scene_info.get("placement_zones", [])
                zone_choices = [zone["name"] for zone in zones]

                # 获取场景中的物体
                objects = scene_info.get("objects", [])
                object_rows = []
                for obj in objects:
                    object_rows.append([
                        Path(obj["original_path"]).name,
                        obj["position"],
                        obj["rotation"],
                        obj["scale"],
                        obj["placement_zone"]
                    ])

                # 场景统计
                stats = {
                    "总物体数": len(objects),
                    "场景尺寸": scene_info.get("dimensions", {}),
                    "放置区域数": len(zones)
                }

                return scene_info, object_rows, stats

            def download_scene(scene_selection):
                if not scene_selection:
                    return "请选择要下载的场景"

                scene_id = scene_selection.split(":")[0]
                success = self.scene_downloader.download_scene(scene_id)

                if success:
                    return f"✅ 场景 '{scene_id}' 下载完成"
                else:
                    return f"❌ 场景 '{scene_id}' 下载失败"

            def place_object(scene_id, object_path, zone, x, y, z, rx, ry, rz, scale, auto_place):
                if not scene_id or not object_path:
                    return "请选择场景和物体"

                if auto_place:
                    # 自动放置
                    success = self.object_placer.place_object(scene_id, object_path, placement_zone=zone)
                else:
                    # 手动放置
                    position = [x, y, z]
                    rotation = [rx, ry, rz]
                    success = self.object_placer.place_object(
                        scene_id, object_path, position, rotation, scale, zone
                    )

                if success:
                    return f"✅ 物体已放置到场景 '{scene_id}' 中"
                else:
                    return f"❌ 放置失败"

            def refresh_scenes():
                scenes = self.object_placer.list_downloaded_scenes()
                return gr.Dropdown(choices=scenes)

            # 绑定事件
            selected_scene.change(
                fn=update_scene_info,
                inputs=[selected_scene],
                outputs=[scene_info_display, scene_objects, scene_stats]
            )

            download_btn.click(
                fn=download_scene,
                inputs=[scene_list],
                outputs=[download_status]
            )

            refresh_btn.click(
                fn=refresh_scenes,
                inputs=[],
                outputs=[selected_scene]
            )

            place_btn.click(
                fn=place_object,
                inputs=[selected_scene, selected_object, placement_zone,
                       pos_x, pos_y, pos_z, rot_x, rot_y, rot_z, scale, auto_place],
                outputs=[place_status]
            ).then(
                fn=update_scene_info,
                inputs=[selected_scene],
                outputs=[scene_info_display, scene_objects, scene_stats]
            )

        return scene_editor

    def _find_generated_objects(self) -> List[str]:
        """查找已生成的3D物体文件"""
        objects = []

        # 查找tmp目录中的GLB文件
        tmp_dir = Path("./tmp")
        if tmp_dir.exists():
            for glb_file in tmp_dir.glob("**/*.glb"):
                objects.append(str(glb_file))

        # 查找当前目录中的GLB文件
        for glb_file in Path(".").glob("*.glb"):
            objects.append(str(glb_file))

        return objects


if __name__ == "__main__":
    editor = SceneEditor()
    app = editor.create_scene_editor_interface()
    app.launch()
