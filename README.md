# TRELLIS Studio

一个基于 `TRELLIS` 深度改造的 3D 资产生成与场景编辑工作台。

它将官方的文本生成 / 图片生成能力整合为一个统一应用，并补充了：

- 中文提示词本地翻译
- 单图 / 多视角 3D 生成
- 生成资产后台管理
- `GLB` / Gaussian `PLY` 导出
- 场景高斯导入与多物体合成
- 基于浏览器的 3DGS 交互式编辑器

> 这是一个偏“应用工作台”的仓库，而不是仅保留官方 demo 的原始镜像。

---

## 功能概览

### 1. 文本生成 3D
- 支持中文 / 英文提示词
- 中文可通过本地翻译模型转英文后生成
- 输出预览视频、`GLB`、Gaussian `PLY`

### 2. 单图生成 3D
- 上传单张参考图生成 3D 资产
- 自动纳入后台资产管理
- 支持继续用于场景编辑

### 3. 多视角融合生成
- 上传 3 张左右多视角图片进行融合生成
- 支持多图模式下的统一导出与资产复用

### 4. 资产管理
- 每次生成的资产自动保存到 `generated_assets/<asset_name>/`
- 自动记录 `metadata.json`
- 自动生成：
  - `preview.mp4`
  - `asset.glb`
  - `asset.ply`
  - `poster.png`

### 5. 场景编辑与合成
- 支持导入场景高斯 `PLY`
- 支持从后台资产中选择物体加入场景
- 支持物体位置、旋转、缩放调整
- 支持多物体 Gaussian 场景合成

### 6. 浏览器交互式编辑器
- 基于 `gs_viewer/` 提供轻量 3DGS 浏览器编辑界面
- 支持物体选择、聚焦、缩放、旋转等操作
- 已针对快捷键与缩放交互做过增强

---

## 项目结构

```text
TRELLIS/
├── app_unified.py              # 主入口：统一 Gradio 工作台
├── app.py                      # 原始官方 image demo
├── app_text.py                 # 原始官方 text demo
├── scene_editor.py             # 场景编辑工作流
├── scene_downloader.py         # 场景导入/管理辅助
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
├── scenes_gs/                  # 场景高斯/合成结果（默认不上传）
├── tmp/                        # 运行临时文件（默认不上传）
└── setup.sh                    # 官方依赖安装脚本
```

---

## 运行环境

### 硬件
- NVIDIA GPU，建议显存 ≥ 16GB

### 系统
- Linux
- Python 3.10 推荐
- Conda 推荐

### 已验证依赖方式
项目保留了官方 `setup.sh` 安装脚本，适合直接基于官方 TRELLIS 环境安装。

---

## 快速开始

### 1. 克隆仓库

```bash
git clone <your-repo-url>
cd TRELLIS
```

### 2. 创建并激活环境

```bash
conda create -n trellis python=3.10 -y
conda activate trellis
```

### 3. 安装依赖

如果你希望沿用官方推荐方式：

```bash
. ./setup.sh --basic --xformers --flash-attn --diffoctreerast --spconv --mipgaussian --kaolin --nvdiffrast
```

如果只想先跑 Web 应用相关依赖，也可以在官方依赖基础上自行补齐所需包。

### 4. 设置环境变量

推荐使用 `xformers`：

```bash
export ATTN_BACKEND=xformers
export SPARSE_ATTN_BACKEND=xformers
export SPCONV_ALGO=native
```

### 5. 准备模型

本项目默认从本地目录读取模型：

- `./models/TRELLIS-image-large`
- `./models/TRELLIS-text-large`
- `./models/translation_zh_en`

你需要提前把这些模型放到对应目录。

#### 必需模型

1. `TRELLIS-image-large`
2. `TRELLIS-text-large`
3. 中文翻译模型 `translation_zh_en`（仅中文文本输入时需要）

#### 目录示例

```text
models/
├── TRELLIS-image-large/
├── TRELLIS-text-large/
└── translation_zh_en/
```

> 注意：这些模型通常体积较大，默认不建议提交到 GitHub。

### 6. 启动统一工作台

```bash
python app_unified.py
```

默认端口：`7860`

如果你要修改端口：

```bash
GRADIO_SERVER_PORT=7861 python app_unified.py
```

启动后在浏览器打开：

```text
http://127.0.0.1:7860
```

---

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
4. 调整位置 / 旋转 / 缩放
5. 生成合成场景并导出
6. 进入浏览器交互编辑器进一步微调

---

## 本地数据目录说明

以下目录会在运行过程中产生内容：

- `generated_assets/`：生成资产结果
- `tmp/`：临时文件
- `scenes_gs/`：场景高斯与合成结果
- `scenes/`：你本地导入的场景文件

这些目录默认都更适合**本地使用**，不建议直接上传到 GitHub。

---

## 开源建议

如果你准备公开这个仓库，建议：

1. **不要上传本地模型**
   - `models/`
   - `openai/`

2. **不要上传生成结果和缓存**
   - `generated_assets/`
   - `tmp/`
   - `scenes_gs/`
   - `*.glb` / `*.ply` / `*.mp4`

3. **不要上传本地环境文件**
   - `.env`
   - `.venv/`

4. **README 里说明模型需要用户自行下载**

本仓库已经通过 `.gitignore` 采用“忽略上传而非删除本地文件”的方式，避免影响你当前环境运行。

---

## 已知注意事项

- `scene_downloader.py` 中的“预设场景下载”部分仍以占位逻辑为主，更适合作为本地导入场景的辅助模块，而不是完整的在线场景库。
- 中文提示词依赖本地翻译模型；如果缺失，只能直接使用英文提示词。
- 场景编辑与 Gaussian 合成更偏工程实验性质，适合本地工作流，不建议直接当作生产级 SaaS 服务使用。

---

## 致谢

本项目基于微软开源的 `TRELLIS` 进行二次开发与工作台化改造。

- Official TRELLIS: https://github.com/microsoft/TRELLIS
- Project Page: https://microsoft.github.io/TRELLIS/

如果你在学术、演示或二次开发中使用了本项目，也建议同时遵守上游仓库的许可证与依赖许可证要求。

---

## License

本仓库当前保留上游 `LICENSE`。如果你准备以个人项目公开发布，建议在确认与上游许可证兼容的前提下继续沿用，并在 README 中明确说明“基于 TRELLIS 二次开发”。
