# TRELLIS Studio

基于 `TRELLIS` 二次开发的 3D 资产生成与场景编辑工作台，提供从生成、预览、导出到场景合成与交互编辑的一体化流程。

## 功能特性

- **文本生成 3D**：支持中文与英文提示词输入，生成 3D 资产并导出结果。
- **单图生成 3D**：上传单张参考图，生成可复用的 3D 资产。
- **多视角融合生成**：支持多张视角图融合，提升生成稳定性与可编辑性。
- **资产管理**：自动保存预览视频、`GLB`、Gaussian `PLY` 与元数据。
- **场景编辑与合成**：支持导入场景高斯、添加物体、调整位姿并导出合成结果。
- **浏览器交互编辑器**：提供基于 `gs_viewer/` 的轻量 3DGS 交互编辑能力。

## 项目结构

```text
TRELLIS/
├── app_unified.py              # 主入口：统一 Gradio 工作台
├── app.py                      # 原始 image demo
├── app_text.py                 # 原始 text demo
├── scene_editor.py             # 场景编辑工作流
├── scene_downloader.py         # 场景导入与管理辅助
├── scene_manager.py            # 场景信息管理
├── object_placer.py            # 物体放置与场景导出
├── gaussian_ply_composer.py    # 多高斯 PLY 合成
├── gs_viewer/                  # 浏览器端 3DGS 查看与交互编辑器
├── trellis/                    # TRELLIS 核心推理/训练代码
├── configs/                    # 官方配置
├── assets/                     # 示例图与静态资源
├── scenes/                     # 场景配置与导入场景
├── models/                     # 本地模型目录（默认不上传）
├── generated_assets/           # 生成结果目录（默认不上传）
├── scenes_gs/                  # 场景高斯与合成结果（默认不上传）
├── tmp/                        # 运行临时文件（默认不上传）
└── setup.sh                    # 依赖安装脚本
```

## 运行环境

- Linux
- Python 3.10
- Conda
- NVIDIA GPU（建议显存 ≥ 16GB）

## 快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/CaiyanC/trellis-studio.git
cd trellis-studio
```

### 2. 创建并激活环境

```bash
conda create -n trellis python=3.10 -y
conda activate trellis
```

### 3. 安装依赖

沿用 TRELLIS 官方安装脚本：

```bash
. ./setup.sh --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast
```

### 4. 设置环境变量

```bash
export ATTN_BACKEND=xformers
export SPARSE_ATTN_BACKEND=xformers
export SPCONV_ALGO=native
```

### 5. 准备模型

项目默认从本地目录读取以下模型：

- `./models/TRELLIS-image-large`
- `./models/TRELLIS-text-large`
- `./models/translation_zh_en`

目录结构示例：

```text
models/
├── TRELLIS-image-large/
├── TRELLIS-text-large/
└── translation_zh_en/
```

> 模型文件通常体积较大，建议本地准备，不纳入仓库版本管理。

### 6. 启动统一工作台

```bash
python app_unified.py
```

默认访问地址：

```text
http://127.0.0.1:7860
```

如需修改端口：

```bash
GRADIO_SERVER_PORT=7861 python app_unified.py
```

## 使用说明

### 文本生成 3D
1. 打开 `📝 文字输入`
2. 输入资产名称
3. 输入提示词
4. 点击生成
5. 导出 `GLB` 或 `PLY`

### 单图生成 3D
1. 打开 `🖼️ 单图片`
2. 上传参考图
3. 输入资产名称
4. 点击生成
5. 在后台资产中复用或导出

### 多视角生成 3D
1. 打开 `📷 多视角`
2. 上传多张视角图
3. 输入资产名称
4. 点击生成

### 场景编辑
1. 打开 `🏗️ 场景编辑器`
2. 选择已有场景高斯，或上传新的场景 `PLY`
3. 从后台资产中选择物体
4. 调整位置、旋转、缩放
5. 生成合成场景并导出
6. 进入浏览器交互编辑器进一步微调

## 本地数据目录说明

运行过程中会生成以下本地目录：

- `generated_assets/`：生成资产结果
- `tmp/`：临时文件
- `scenes_gs/`：场景高斯与合成结果
- `scenes/`：本地导入的场景文件

这些目录主要用于本地工作流，不建议直接纳入公开仓库。

## 已知注意事项

- `scene_downloader.py` 中的预设场景下载逻辑目前以占位实现为主，更适合作为本地导入场景的辅助模块。
- 中文提示词依赖本地翻译模型；若缺失，可直接使用英文提示词。
- 场景编辑与 Gaussian 合成功能更偏本地工作流与工程实验场景。

## 致谢

本项目基于微软开源的 `TRELLIS` 进行二次开发与工作台化改造。

- TRELLIS: https://github.com/microsoft/TRELLIS
- Project Page: https://microsoft.github.io/TRELLIS/

## License

本仓库保留上游 `LICENSE`，并在其基础上进行二次开发与工作台化改造。使用本项目时，请同时遵守上游仓库及相关依赖的许可证要求。
