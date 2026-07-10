# SAM3 - Segment Anything Model 3

基于 Meta AI 的 SAM3 实现，支持文本提示的图像分割。

## 功能

- **单图分割** (`sam3_run.py`) — 输入图片 + 文本提示词，输出分割掩码
- **HTTP 服务** (`http_sam3.py`) — Flask API，支持远程调用
- **客户端示例** (`use_http_sam3.py`) — 调用 HTTP 服务的示例代码

## 环境要求

- Python >= 3.8
- PyTorch（CUDA 推荐）
- 依赖见 `pyproject.toml`

## 安装

```bash
# 克隆仓库
git clone https://github.com/suiyuanddie/sam3.git
cd sam3

# 安装依赖
pip install -e .

# 额外依赖（用于 HTTP 服务）
pip install flask rich opencv-python
```

## 模型权重下载

`sam3.pt` 模型文件（约 3.2GB）未包含在仓库中，请从以下地址下载并放置于项目根目录：

> **下载地址**: [待补充 - 请提供模型下载链接]

下载后目录结构应为：
```
SAM3/
├── sam3.pt          ← 模型权重文件（需自行下载）
├── sam3_run.py
├── http_sam3.py
├── use_http_sam3.py
├── input/           ← 放入待分割的图片
└── output/          ← 分割结果输出目录
```

## 使用方法

### 1. 单图分割

```bash
# 将图片放入 input/ 文件夹，然后运行
python sam3_run.py

# 或直接指定提示词
python sam3_run.py "cup, bowl, banana"
```

### 2. HTTP 服务

```bash
# 启动服务（默认 http://127.0.0.1:5001）
python http_sam3.py

# 调用示例
python use_http_sam3.py
```

#### API 接口

**POST `/process`**

| 参数 | 类型 | 说明 |
|------|------|------|
| `image` | file | 输入图片 |
| `text_prompt` | string | 提示词，逗号分隔 |
| `compact` | string | `"true"` 返回 URL 而非 base64 |

返回检测结果图、合并掩码和单个物体掩码。

**GET `/image/<request_id>/detection`** — 下载检测结果图  
**GET `/image/<request_id>/merged`** — 下载合并掩码  
**GET `/image/<request_id>/mask/<id>`** — 下载单个掩码

## 项目结构

```
sam3/
├── sam3/                  # 核心库
│   ├── model/             # 模型定义
│   ├── agent/             # Agent 推理逻辑
│   ├── sam/               # SAM 组件（mask decoder, prompt encoder 等）
│   ├── train/             # 训练相关代码
│   ├── eval/              # 评估工具
│   └── perflib/           # 性能优化库
├── sam3_run.py            # 单图分割脚本
├── http_sam3.py           # HTTP API 服务
├── use_http_sam3.py       # HTTP 客户端示例
├── input/                 # 输入图片目录
├── output/                # 输出结果目录
└── pyproject.toml         # 项目配置
```

## License

MIT License
