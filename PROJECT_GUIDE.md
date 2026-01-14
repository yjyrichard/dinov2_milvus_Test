# DINOv2 图像检索系统 - 完整技术指南

> 本文档将手把手教你理解这个项目的每一个细节，从架构设计到代码实现。

---

## 目录

1. [项目概述](#1-项目概述)
2. [技术栈总览](#2-技术栈总览)
3. [项目架构详解](#3-项目架构详解)
4. [FastAPI 框架详解](#4-fastapi-框架详解)
5. [图像处理技术详解](#5-图像处理技术详解)
6. [DINOv2 模型详解](#6-dinov2-模型详解)
7. [Milvus 向量数据库详解](#7-milvus-向量数据库详解)
8. [MinIO 对象存储详解](#8-minio-对象存储详解)
9. [完整数据流程分析](#9-完整数据流程分析)
10. [外观专利数据切分与导入策略](#10-外观专利数据切分与导入策略)
11. [设计模式与最佳实践](#11-设计模式与最佳实践)
12. [生产环境注意事项](#12-生产环境注意事项)

---

## 1. 项目概述

### 1.1 这个项目是做什么的？

这是一个**以图搜图**系统，核心功能是：
- 用户上传一张图片
- 系统找出数据库中与之最相似的图片
- 返回相似度排序的结果

### 1.2 应用场景

- **专利检索**：查找相似的外观专利设计
- **图片去重**：找出重复或相似的图片
- **商品搜索**：电商平台的相似商品推荐

### 1.3 核心原理（一句话总结）

```
图片 → DINOv2模型 → 384/768维向量 → Milvus向量检索 → 相似图片
```

---

## 2. 技术栈总览

| 层级 | 技术 | 作用 | 为什么选它 |
|------|------|------|------------|
| Web框架 | FastAPI | HTTP API 服务 | 高性能、异步、自动生成文档 |
| 深度学习 | PyTorch + Transformers | 运行 DINOv2 模型 | 业界标准 |
| 视觉模型 | DINOv2 | 图像特征提取 | Meta AI 开源，效果好 |
| 向量数据库 | Milvus | 相似度检索 | 专为向量设计，速度快 |
| 图像处理 | Pillow (PIL) | 图片读取、缩放 | Python 图像处理标准库 |
| 对象存储 | MinIO | 存储原始图片 | S3 兼容，私有化部署 |

---

## 3. 项目架构详解

### 3.1 目录结构

```
dinov2_milvus_Test/
├── main.py                    # 应用入口（重要）
├── requirements.txt           # Python 依赖
│
├── app/                       # 应用核心代码
│   ├── __init__.py
│   ├── config.py              # 全局配置（重要）
│   │
│   ├── api/                   # API 路由层（Controller）
│   │   ├── search.py          # 搜索接口
│   │   ├── batch.py           # 批量导入接口
│   │   └── image.py           # 图片服务接口
│   │
│   └── services/              # 业务逻辑层（Service）
│       ├── dinov2_service.py  # DINOv2 特征提取
│       ├── milvus_service.py  # Milvus 数据库操作
│       ├── minio_service.py   # MinIO 文件存储
│       ├── image_preprocessor.py  # 图像预处理
│       └── batch_import_service.py # 批量导入逻辑
│
├── scripts/                   # 独立运行的脚本
│   └── import_design_patents_full.py
│
└── web/                       # 前端项目（Vue 3）
    └── dinov2TestWeb/
```

### 3.2 分层架构图

```
┌─────────────────────────────────────────────────────────────┐
│                        前端 (Vue 3)                          │
│                  用户界面、图片上传、结果展示                   │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTP 请求
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     API 层 (app/api/)                        │
│   接收请求、参数校验、调用服务层、返回响应                       │
│   search.py | batch.py | image.py                           │
└───────────────────────────┬─────────────────────────────────┘
                            │ 调用
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                   服务层 (app/services/)                      │
│   核心业务逻辑：特征提取、向量检索、文件存储                     │
│   dinov2_service | milvus_service | minio_service            │
└───────────────────────────┬─────────────────────────────────┘
                            │ 调用
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      外部服务/基础设施                         │
│   PyTorch (GPU) | Milvus 集群 | MinIO 集群                   │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 为什么要分层？

| 层级 | 职责 | 好处 |
|------|------|------|
| API 层 | 处理 HTTP 请求/响应 | 接口变化不影响业务逻辑 |
| Service 层 | 实现业务逻辑 | 可以被多个 API 复用 |
| 外部服务 | 数据存储、计算 | 可以独立扩展 |

---

## 4. FastAPI 框架详解

### 4.1 什么是 FastAPI？

FastAPI 是一个现代 Python Web 框架，特点：
- **高性能**：基于 Starlette 和 Pydantic
- **异步支持**：原生支持 async/await
- **自动文档**：访问 `/docs` 即可看到 Swagger UI

### 4.2 应用入口 (main.py) 详解

```python
# main.py 核心代码解析

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ============ 生命周期管理 ============
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理器
    - 启动时执行：yield 之前的代码
    - 关闭时执行：yield 之后的代码
    """
    # 启动时：初始化 Milvus 连接
    print("[APP] Starting...")

    # 在后台线程中初始化，不阻塞 API 服务
    init_thread = threading.Thread(target=init_milvus_in_background, daemon=True)
    init_thread.start()

    yield  # 应用运行中...

    # 关闭时：清理资源
    print("[APP] Shutting down...")


# ============ 创建应用实例 ============
app = FastAPI(
    title="DINOv2 Image Search",        # 显示在文档中的标题
    description="基于 DINOv2 的图像相似度检索系统",  # 描述
    version="1.0.0",                    # 版本号
    lifespan=lifespan                   # 生命周期管理器
)


# ============ CORS 中间件 ============
# CORS = Cross-Origin Resource Sharing (跨域资源共享)
# 允许前端从不同域名访问后端 API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # 允许所有来源（生产环境应限制）
    allow_credentials=True,   # 允许携带 Cookie
    allow_methods=["*"],      # 允许所有 HTTP 方法
    allow_headers=["*"],      # 允许所有请求头
)


# ============ 注册路由 ============
# 将 api 模块中的路由注册到应用
from app.api import search, batch, image

app.include_router(search.router)   # /api/search 等
app.include_router(batch.router)    # /api/batch/start 等
app.include_router(image.router)    # /api/image/{file_name} 等
```

### 4.3 路由定义详解 (search.py)

```python
# app/api/search.py 核心代码解析

from fastapi import APIRouter, UploadFile, File, Form

# ============ 创建路由器 ============
router = APIRouter(
    prefix="/api",     # 所有路由的前缀
    tags=["search"]    # Swagger 文档中的分组标签
)


# ============ POST 接口：以图搜图 ============
@router.post("/search")
async def search_image(
    # 参数说明：
    file: UploadFile = File(...),           # 上传的文件，... 表示必填
    top_k: int = Form(DEFAULT_TOP_K),       # 表单字段，返回前 K 个结果
    min_score: float = Form(DEFAULT_MIN_SCORE),  # 最小相似度阈值
):
    """
    以图搜图接口

    这是一个 POST 请求，接收 multipart/form-data 格式的数据
    - file: 图片文件
    - top_k: 返回数量
    - min_score: 相似度阈值
    """
    # 1. 读取上传的图片
    contents = await file.read()  # async 异步读取
    image = Image.open(io.BytesIO(contents))

    # 2. 提取特征向量
    query_vector, feature_time = dinov2_extractor.extract_from_pil(image)

    # 3. 在 Milvus 中搜索
    results, search_time = milvus_service.search(
        query_vector=query_vector,
        top_k=top_k,
        min_score=min_score
    )

    # 4. 返回 JSON 响应
    return {
        "code": 0,
        "message": "success",
        "data": {
            "results": results,
            "timing": {...}
        }
    }


# ============ GET 接口：获取统计信息 ============
@router.get("/collection/stats")
async def get_collection_stats():
    """获取 Collection 统计信息"""
    stats = milvus_service.get_stats()
    return {"code": 0, "data": stats}
```

### 4.4 FastAPI 关键概念速查

| 概念 | 说明 | 代码示例 |
|------|------|----------|
| 路由 | URL 路径映射 | `@router.get("/path")` |
| 路径参数 | URL 中的变量 | `/image/{file_name}` |
| 查询参数 | URL ? 后的参数 | `?top_k=10` |
| 表单数据 | POST 表单字段 | `Form(...)` |
| 文件上传 | 上传文件 | `File(...)` |
| 响应模型 | 规范返回格式 | `response_model=XXX` |
| 中间件 | 请求/响应拦截 | `app.add_middleware(...)` |

---

## 5. 图像处理技术详解

### 5.1 使用的库：Pillow (PIL)

```python
from PIL import Image
```

Pillow 是 Python 最常用的图像处理库，注意：
- **这个项目没有使用 OpenCV**，纯粹使用 Pillow
- Pillow 更轻量，安装简单，足够满足需求

### 5.2 Pillow 核心操作详解

```python
# ============ 1. 打开图片 ============
img = Image.open("path/to/image.jpg")
# 也可以从内存中打开
img = Image.open(io.BytesIO(binary_data))


# ============ 2. 获取图片信息 ============
width, height = img.size  # 尺寸
mode = img.mode           # 颜色模式: "RGB", "RGBA", "L" (灰度) 等


# ============ 3. 颜色模式转换 ============
# TIF 文件可能是 CMYK 或其他模式，需要转为 RGB
img = img.convert("RGB")


# ============ 4. 缩放图片 ============
# resize: 强制缩放到指定尺寸（可能变形）
img = img.resize((518, 518), Image.Resampling.LANCZOS)

# thumbnail: 等比例缩放，不超过指定尺寸（不会变形）
img.thumbnail((300, 300), Image.Resampling.LANCZOS)


# ============ 5. 创建空白图片 ============
# 创建白色背景
background = Image.new("RGB", (518, 518), (255, 255, 255))
#                       模式    尺寸          颜色(R,G,B)


# ============ 6. 粘贴图片 ============
# 把 img 粘贴到 background 的 (x, y) 位置
background.paste(img, (paste_x, paste_y))


# ============ 7. 保存图片 ============
img.save("output.jpg", "JPEG", quality=85)
#        文件路径      格式     质量(1-100)

# 保存到内存
buffer = io.BytesIO()
img.save(buffer, format="JPEG", quality=85)
buffer.seek(0)  # 重置指针到开头，便于读取


# ============ 8. 处理多页 TIF ============
# TIF 文件可能包含多页
if hasattr(img, "n_frames") and img.n_frames > 1:
    img.seek(0)  # 跳转到第一页
```

### 5.3 Letterbox Padding 技术详解

**问题**：DINOv2 模型要求输入 518x518 的正方形图片，但原图可能是各种尺寸

**方案对比**：

| 方案 | 做法 | 问题 |
|------|------|------|
| 直接 Resize | 强制缩放到 518x518 | 长方形图片被拉伸变形 |
| 中心裁剪 | 裁剪出中间部分 | 丢失边缘信息 |
| **Letterbox Padding** | 等比缩放 + 白色填充 | 保持原始比例，无信息损失 |

**Letterbox 实现代码详解**：

```python
# app/services/image_preprocessor.py

def letterbox_resize(image: Image.Image, target_size: int = 518) -> Image.Image:
    """
    Letterbox Padding 预处理

    例如：1000x500 的图片
    1. 计算缩放比例：518/1000 = 0.518
    2. 缩放后尺寸：1000*0.518=518, 500*0.518=259 → 518x259
    3. 创建 518x518 白色背景
    4. 把 518x259 居中贴上，上下各留空 (518-259)/2=129.5 ≈ 129 像素
    """

    # 确保是 RGB 模式
    if image.mode != "RGB":
        image = image.convert("RGB")

    # 获取原始尺寸
    orig_width, orig_height = image.size  # 例如 1000, 500

    # 计算缩放比例：长边缩放到 target_size
    scale = target_size / max(orig_width, orig_height)
    # scale = 518 / 1000 = 0.518

    # 计算缩放后的尺寸
    new_width = int(orig_width * scale)   # 518
    new_height = int(orig_height * scale) # 259

    # 高质量缩放（LANCZOS 是最高质量的重采样算法）
    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # 创建白色背景
    background = Image.new("RGB", (target_size, target_size), (255, 255, 255))

    # 计算居中粘贴位置
    paste_x = (target_size - new_width) // 2   # (518-518)//2 = 0
    paste_y = (target_size - new_height) // 2  # (518-259)//2 = 129

    # 粘贴到背景上
    background.paste(resized, (paste_x, paste_y))

    return background
```

**可视化效果**：

```
原图 (1000x500):          Letterbox 后 (518x518):
┌────────────────┐        ┌──────────────┐
│                │        │ (白色填充)    │
│   图片内容      │   →    ├──────────────┤
│                │        │   图片内容    │
└────────────────┘        ├──────────────┤
                          │ (白色填充)    │
                          └──────────────┘
```

### 5.4 缩略图生成

```python
def generate_thumbnail(src_path: str, size: tuple[int, int] = (300, 300)):
    """生成缩略图用于前端预览"""

    img = Image.open(src_path)

    # 处理多页 TIF
    if hasattr(img, "n_frames") and img.n_frames > 1:
        img.seek(0)

    # 转为 RGB
    img = img.convert("RGB")

    # thumbnail 是等比例缩放，不会变形
    # 最终尺寸 ≤ (300, 300)
    img.thumbnail(size, Image.Resampling.LANCZOS)

    # 保存为 JPEG
    img.save(dst_path, "JPEG", quality=85)
```

---

## 6. DINOv2 模型详解

### 6.1 什么是 DINOv2？

**DINOv2** (DIstillation with NO labels v2) 是 Meta AI 在 2023 年发布的视觉基础模型。

**核心特点**：
- **自监督学习**：不需要人工标注数据
- **通用特征**：提取的特征可用于各种下游任务
- **高质量**：在 ImageNet 等数据集上训练，效果优秀

### 6.2 DINOv2 的工作原理

```
输入图片 (518x518)
    │
    ▼
┌─────────────────────────────────┐
│   DINOv2 Vision Transformer     │
│   (ViT 架构)                    │
│                                 │
│   1. 将图片切成 14x14 个 patch  │
│   2. 每个 patch 编码为向量      │
│   3. 加上位置编码               │
│   4. 通过 Transformer 处理      │
└─────────────────────────────────┘
    │
    ▼
输出: [CLS] token + 196 个 patch tokens
       │
       └── 我们只用 CLS token (384/768 维)
           这个向量代表整张图片的语义特征
```

### 6.3 模型变体

| 模型 | 参数量 | 向量维度 | 速度 | 效果 |
|------|--------|----------|------|------|
| dinov2-small | 22M | 384 | 快 | 一般 |
| dinov2-base | 86M | 768 | 中等 | 较好 |
| dinov2-large | 300M | 1024 | 慢 | 很好 |
| dinov2-giant | 1.1B | 1536 | 很慢 | 最好 |

本项目使用 **small** 和 **base** 两个模型。

### 6.4 代码实现详解

```python
# app/services/dinov2_service.py

import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image

class DINOv2Extractor:
    """DINOv2 特征提取器"""

    # ============ 单例模式 ============
    _instance = None

    def __new__(cls):
        """
        单例模式：整个应用只有一个实例
        避免重复加载模型浪费内存
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance


    # ============ 延迟初始化 ============
    def _ensure_initialized(self):
        """
        延迟初始化：首次使用时才加载模型
        好处：应用启动更快，内存按需使用
        """
        if self._initialized:
            return

        # 检测设备：有 GPU 用 GPU，否则用 CPU
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # 加载图像处理器（负责预处理：归一化、标准化等）
        self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")

        # 加载模型
        self.model = AutoModel.from_pretrained("facebook/dinov2-small")
        self.model.to(self.device)  # 移动到 GPU/CPU
        self.model.eval()           # 设为推理模式（关闭 dropout 等）

        self._initialized = True


    # ============ 特征提取 ============
    def extract_from_pil(self, image: Image.Image) -> tuple[list[float], float]:
        """
        从 PIL Image 提取特征向量

        参数：
            image: PIL 图片对象
        返回：
            (384维向量, 耗时毫秒)
        """
        self._ensure_initialized()

        start_time = time.time()

        # 1. 颜色模式转换
        image = image.convert("RGB")

        # 2. Letterbox Padding 预处理
        image = preprocess_for_dinov2(image)  # 变成 518x518

        # 3. 使用 processor 进行标准化
        # processor 会：
        # - 转换为 tensor
        # - 归一化到 [0, 1]
        # - 减均值除方差（ImageNet 标准化）
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = inputs.to(self.device)  # 移到 GPU

        # 4. 模型推理
        with torch.no_grad():  # 关闭梯度计算，节省内存
            outputs = self.model(**inputs)

            # outputs.last_hidden_state 的 shape: [batch_size, num_tokens, hidden_dim]
            # 例如: [1, 197, 384]
            # - 197 = 1 (CLS token) + 196 (14x14 patch tokens)
            # - 384 = 隐藏层维度

            # 取 CLS token（第 0 个 token）
            features = outputs.last_hidden_state[:, 0, :]  # shape: [1, 384]

            # 5. L2 归一化
            # 使向量长度为 1，便于用余弦相似度比较
            features = torch.nn.functional.normalize(features, p=2, dim=1)

        elapsed_ms = (time.time() - start_time) * 1000

        # 转换为 Python list 返回
        return features.squeeze().cpu().tolist(), elapsed_ms
```

### 6.5 关键概念解释

#### CLS Token 是什么？

```
Transformer 的输入序列:
[CLS] [patch1] [patch2] ... [patch196]
  │
  └── CLS (Class) token 是一个特殊的学习向量
      它通过 attention 机制聚合了所有 patch 的信息
      最终代表整张图片的全局语义特征
```

#### L2 归一化是什么？

```python
# L2 归一化：把向量缩放到单位长度
# 原理：向量 / 向量的 L2 范数

import numpy as np

vector = np.array([3, 4])  # 原向量
l2_norm = np.sqrt(3**2 + 4**2)  # = 5
normalized = vector / l2_norm  # = [0.6, 0.8]

# 归一化后向量长度为 1
np.sqrt(0.6**2 + 0.8**2)  # = 1.0

# 好处：两个归一化向量的点积 = 余弦相似度
# cos(θ) = A·B / (|A||B|) = A·B (因为 |A|=|B|=1)
```

#### 为什么用余弦相似度？

```
余弦相似度衡量的是向量的"方向"而非"大小"
- 两个向量方向相同：cos = 1
- 两个向量垂直：cos = 0
- 两个向量方向相反：cos = -1

这对于语义相似度很有意义：
- 即使两张图片亮度不同，特征向量的方向应该相似
- 我们关心的是"这是什么"，而不是"这有多亮"
```

---

## 7. Milvus 向量数据库详解

### 7.1 什么是向量数据库？

**传统数据库**：
```sql
SELECT * FROM products WHERE category = 'electronics' AND price < 100
-- 精确匹配或范围查询
```

**向量数据库**：
```python
# 找出与 query_vector 最相似的 10 个向量
results = collection.search(data=[query_vector], limit=10)
-- 相似度/距离查询
```

### 7.2 Milvus 核心概念

| 概念 | 传统数据库类比 | 说明 |
|------|----------------|------|
| Collection | Table | 存储数据的容器 |
| Entity | Row | 一条数据记录 |
| Field | Column | 数据字段 |
| Index | Index | 加速搜索的数据结构 |
| Partition | Partition | 数据分区 |

### 7.3 代码实现详解

```python
# app/services/milvus_service.py

from pymilvus import (
    connections,       # 连接管理
    Collection,        # Collection 操作
    FieldSchema,       # 字段定义
    CollectionSchema,  # Schema 定义
    DataType,          # 数据类型
    utility            # 工具函数
)


class MilvusService:
    """Milvus 向量数据库服务"""

    # ============ 连接 Milvus ============
    def connect(self) -> bool:
        """
        建立与 Milvus 服务器的连接

        Milvus 部署方式：
        - Standalone: 单机版，适合开发测试
        - Cluster: 集群版，适合生产环境
        """
        connections.connect(
            alias="default",          # 连接别名
            host="192.168.1.174",     # Milvus 服务器地址
            port="31278"              # 端口号
        )
        self._connected = True
        return True


    # ============ 创建 Collection ============
    def create_collection(self) -> Collection:
        """
        创建 Collection（类似创建数据库表）

        定义数据结构：
        - 主键字段
        - 普通字段
        - 向量字段
        """

        # 定义字段
        fields = [
            # 主键字段：自增 ID
            FieldSchema(
                name="id",
                dtype=DataType.INT64,
                is_primary=True,      # 设为主键
                auto_id=True          # 自动生成
            ),

            # 普通字段：存储元数据
            FieldSchema(
                name="file_name",
                dtype=DataType.VARCHAR,
                max_length=256        # VARCHAR 需要指定最大长度
            ),
            FieldSchema(
                name="patent_id",
                dtype=DataType.VARCHAR,
                max_length=64
            ),
            FieldSchema(
                name="page_num",
                dtype=DataType.VARCHAR,
                max_length=16
            ),
            FieldSchema(
                name="file_path",
                dtype=DataType.VARCHAR,
                max_length=512
            ),

            # 向量字段：存储 DINOv2 特征
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=384               # 向量维度，必须与模型输出一致
            ),

            # 时间戳字段
            FieldSchema(
                name="created_at",
                dtype=DataType.INT64
            ),
        ]

        # 创建 Schema
        schema = CollectionSchema(
            fields,
            description="Patent image embeddings"
        )

        # 创建 Collection
        self._collection = Collection(
            name="patent_images",
            schema=schema
        )

        return self._collection


    # ============ 创建索引 ============
    def create_index(self) -> bool:
        """
        创建向量索引（加速搜索的关键）

        索引类型说明：
        - FLAT: 暴力搜索，100% 精确，但慢
        - IVF_FLAT: 倒排索引 + 暴力，速度快，精度高
        - IVF_SQ8: IVF + 标量量化，更省内存
        - HNSW: 图索引，查询很快，占用内存多
        """

        index_params = {
            "index_type": "IVF_FLAT",  # 索引类型
            "metric_type": "COSINE",   # 距离度量：余弦相似度
            "params": {
                "nlist": 128           # 聚类中心数量
            }
        }

        self._collection.create_index(
            field_name="embedding",    # 对哪个字段建索引
            index_params=index_params
        )

        return True


    # ============ 加载 Collection ============
    def load_collection(self) -> bool:
        """
        将 Collection 加载到内存

        Milvus 的存储分层：
        - 磁盘：持久化存储
        - 内存：搜索时需要

        必须 load 之后才能搜索！
        """
        self._collection.load()
        return True


    # ============ 插入数据 ============
    def insert(self, data: list[dict]) -> list[int]:
        """
        插入数据到 Collection

        参数：
            data: [
                {
                    "file_name": "xxx.tif",
                    "patent_id": "USD123456",
                    "page_num": "D00001",
                    "file_path": "/path/to/file",
                    "embedding": [0.1, 0.2, ...]  # 384维向量
                },
                ...
            ]
        """
        collection = self.get_collection()

        # Milvus 期望的格式：按字段组织
        entities = [
            [d["file_name"] for d in data],    # 所有 file_name
            [d["patent_id"] for d in data],    # 所有 patent_id
            [d["page_num"] for d in data],     # 所有 page_num
            [d["file_path"] for d in data],    # 所有 file_path
            [d["embedding"] for d in data],    # 所有 embedding
            [int(time.time()) for _ in data],  # 所有 created_at
        ]

        # 执行插入
        result = collection.insert(entities)

        # flush 确保数据持久化到磁盘
        collection.flush()

        # 返回插入记录的 ID
        return result.primary_keys


    # ============ 搜索相似向量 ============
    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        min_score: float = 0.5
    ) -> tuple[list[dict], float]:
        """
        搜索与 query_vector 最相似的向量

        参数：
            query_vector: 查询向量 (384维)
            top_k: 返回前 K 个结果
            min_score: 最小相似度阈值

        返回：
            (结果列表, 搜索耗时ms)
        """
        collection = self.get_collection()

        search_params = {
            "metric_type": "COSINE",   # 余弦相似度
            "params": {
                "nprobe": 16           # 搜索时探测的聚类数量
            }
        }

        # 执行搜索
        results = collection.search(
            data=[query_vector],       # 查询向量（可以批量查询）
            anns_field="embedding",    # 在哪个字段上搜索
            param=search_params,       # 搜索参数
            limit=top_k * 2,           # 多取一些，后续过滤
            output_fields=[            # 返回哪些字段
                "file_name", "patent_id", "page_num", "file_path"
            ]
        )

        # 过滤低于阈值的结果
        filtered = []
        for hit in results[0]:  # results[0] 对应第一个查询向量
            if hit.score >= min_score:
                filtered.append({
                    "id": hit.id,
                    "score": float(hit.score),  # 相似度分数
                    "file_name": hit.entity.get("file_name"),
                    "patent_id": hit.entity.get("patent_id"),
                    "page_num": hit.entity.get("page_num"),
                    "file_path": hit.entity.get("file_path"),
                })

        return filtered[:top_k], search_time_ms
```

### 7.4 索引参数详解

#### IVF_FLAT 索引原理

```
原理：倒排文件索引 (Inverted File Index)

1. 训练阶段（创建索引时）：
   - 对所有向量进行 K-means 聚类
   - nlist=128 表示分成 128 个簇

2. 插入阶段：
   - 每个向量被分配到最近的簇

3. 搜索阶段：
   - 先找到查询向量最近的 nprobe 个簇
   - 只在这些簇内做暴力搜索

速度提升：
   - 原本需要搜索 100 万个向量
   - 现在只需搜索 100万/128*16 ≈ 12.5 万个向量
   - 速度提升约 8 倍
```

#### 参数选择指南

| 参数 | 含义 | 小值 | 大值 |
|------|------|------|------|
| nlist | 聚类数量 | 搜索快但精度低 | 搜索慢但精度高 |
| nprobe | 探测簇数 | 搜索快但精度低 | 搜索慢但精度高 |

推荐值：
- 数据量 < 100 万：`nlist=128, nprobe=16`
- 数据量 100万-1000万：`nlist=1024, nprobe=64`

---

## 8. MinIO 对象存储详解

### 8.1 什么是 MinIO？

MinIO 是一个高性能的对象存储系统：
- **兼容 S3 API**：可以用 AWS S3 的 SDK 访问
- **私有化部署**：数据存在自己服务器
- **高性能**：专为大文件设计

### 8.2 核心概念

| MinIO 概念 | 传统文件系统类比 | 说明 |
|------------|------------------|------|
| Bucket | 根目录 | 存储容器 |
| Object | 文件 | 存储的实际数据 |
| Object Key | 文件路径 | 对象的唯一标识 |

### 8.3 代码实现

```python
# app/services/minio_service.py

from minio import Minio
import io


class MinIOService:
    """MinIO 对象存储服务"""

    def __init__(self):
        self.client = None


    def connect(self):
        """连接 MinIO"""
        if self.client:
            return

        self.client = Minio(
            endpoint="192.168.1.174:9000",  # MinIO 服务器地址
            access_key="admin",              # 访问密钥
            secret_key="trizhi2026",         # 密钥
            secure=False                     # 使用 HTTP（生产环境应用 HTTPS）
        )


    def upload_file(self, file_path: str, object_name: str) -> bool:
        """
        上传文件到 MinIO

        参数：
            file_path: 本地文件路径
            object_name: 在 MinIO 中的路径，如 "images/test.jpg"
        """
        self.connect()

        self.client.fput_object(
            bucket_name="trizhi01",    # Bucket 名称
            object_name=object_name,   # 对象路径
            file_path=file_path        # 本地文件
        )
        return True


    def upload_bytes(self, data: bytes, object_name: str, content_type: str) -> bool:
        """
        上传字节数据到 MinIO

        用于直接上传内存中的数据
        """
        self.connect()

        data_stream = io.BytesIO(data)
        data_length = len(data)

        self.client.put_object(
            bucket_name="trizhi01",
            object_name=object_name,
            data=data_stream,
            length=data_length,
            content_type=content_type  # 如 "image/jpeg"
        )
        return True


    def download_file(self, object_name: str) -> bytes | None:
        """
        从 MinIO 下载文件

        返回文件的字节数据
        """
        self.connect()

        try:
            response = self.client.get_object(
                bucket_name="trizhi01",
                object_name=object_name
            )
            return response.read()
        except Exception as e:
            print(f"Download failed: {e}")
            return None
        finally:
            response.close()
            response.release_conn()


    def file_exists(self, object_name: str) -> bool:
        """检查文件是否存在"""
        self.connect()

        try:
            self.client.stat_object("trizhi01", object_name)
            return True
        except:
            return False


    def get_url(self, object_name: str) -> str:
        """
        生成文件的访问 URL

        注意：这是内部 URL，可能需要配置代理才能外部访问
        """
        return f"http://192.168.1.174:9000/trizhi01/{object_name}"
```

---

## 9. 完整数据流程分析

### 9.1 批量导入流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                        批量导入流程                                  │
└─────────────────────────────────────────────────────────────────────┘

1. 用户调用 POST /api/batch/start
   │
   ▼
2. 扫描图片目录
   │  os.listdir(IMAGE_DIR)
   │  过滤支持的格式：.tif, .jpg, .png 等
   │
   ▼
3. 检查已存在的文件（断点续传）
   │  milvus_service.get_existing_files()
   │  跳过已导入的文件
   │
   ▼
4. 分批处理（BATCH_SIZE=32）
   │
   │  ┌─────────────────────────────────────────────┐
   │  │  每批处理：                                  │
   │  │                                             │
   │  │  4.1 加载图片                               │
   │  │      Image.open(path)                      │
   │  │      ↓                                     │
   │  │  4.2 Letterbox Padding                     │
   │  │      preprocess_for_dinov2(image)          │
   │  │      ↓                                     │
   │  │  4.3 DINOv2 特征提取                       │
   │  │      dinov2_extractor.extract_batch()      │
   │  │      输出：32 个 384 维向量                 │
   │  │      ↓                                     │
   │  │  4.4 解析文件名                            │
   │  │      parse_filename() → patent_id, page_num│
   │  │      ↓                                     │
   │  │  4.5 生成缩略图                            │
   │  │      generate_thumbnail() → 300x300 JPEG   │
   │  │      ↓                                     │
   │  │  4.6 插入 Milvus                           │
   │  │      milvus_service.insert(data)           │
   │  │                                             │
   │  └─────────────────────────────────────────────┘
   │
   ▼
5. 保存进度到 JSON 文件
   │  import_progress.json
   │
   ▼
6. 返回导入状态
```

### 9.2 以图搜图流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                       以图搜图流程                                   │
└─────────────────────────────────────────────────────────────────────┘

1. 用户上传图片
   │  POST /api/search
   │  Content-Type: multipart/form-data
   │  - file: 图片文件
   │  - top_k: 10
   │  - min_score: 0.4
   │
   ▼
2. FastAPI 接收请求
   │  async def search_image(file: UploadFile, ...)
   │
   ▼
3. 读取图片到内存
   │  contents = await file.read()
   │  image = Image.open(io.BytesIO(contents))
   │
   ▼
4. 图像预处理
   │  image = image.convert("RGB")      # 转为 RGB
   │  image = preprocess_for_dinov2()   # Letterbox 到 518x518
   │
   ▼
5. DINOv2 特征提取
   │  inputs = processor(images=image, return_tensors="pt")
   │  outputs = model(**inputs)
   │  features = outputs.last_hidden_state[:, 0, :]  # CLS token
   │  features = normalize(features)                  # L2 归一化
   │  query_vector = features.tolist()               # 转为 Python list
   │
   │  耗时：约 50-100ms (GPU) / 500-1000ms (CPU)
   │
   ▼
6. Milvus 向量检索
   │  results = collection.search(
   │      data=[query_vector],
   │      anns_field="embedding",
   │      limit=top_k * 2,
   │      output_fields=[...]
   │  )
   │
   │  耗时：约 5-20ms
   │
   ▼
7. 后处理
   │  7.1 过滤低分结果 (score < min_score)
   │  7.2 按专利号归组
   │  7.3 计算每组最高分
   │  7.4 按最高分排序
   │
   ▼
8. 返回 JSON 响应
   │  {
   │    "code": 0,
   │    "data": {
   │      "results": [...],
   │      "timing": {
   │        "feature_extraction_ms": 80,
   │        "milvus_search_ms": 12,
   │        "total_ms": 95
   │      }
   │    }
   │  }
   │
   ▼
9. 前端展示结果
   │  - 显示缩略图
   │  - 显示相似度分数
   │  - 点击查看原图
```

### 9.3 图片服务流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                       图片服务流程                                   │
└─────────────────────────────────────────────────────────────────────┘

GET /api/image/{file_name}  (获取缩略图)
│
├─→ 1. 检查本地缩略图缓存
│       thumbnails/{name}.jpg
│       │
│       ├── 存在 → 直接返回 FileResponse
│       │
│       └── 不存在 → 继续
│
├─→ 2. 查找原图
│       │
│       ├── 本地目录 (testImage/, 睿观/)
│       │   Image.open(path)
│       │
│       └── MinIO
│           minio_service.download_file()
│           Image.open(io.BytesIO(data))
│
├─→ 3. 找不到 → 返回 404
│
└─→ 4. 实时生成缩略图
        img.thumbnail((300, 300))
        img.save(buffer, "JPEG", quality=85)
        return StreamingResponse(buffer)
```

---

## 10. 外观专利数据切分与导入策略

### 10.1 USPTO 外观专利数据结构

你当前处理的是 USPTO（美国专利商标局）的外观专利数据，其目录结构如下：

```
data/
├── USD1107373/                    # 每个专利一个目录
│   ├── USD1107373-20251230.XML    # 元数据文件（XML格式）
│   ├── USD1107373-20251230-D00000.TIF  # 封面图
│   ├── USD1107373-20251230-D00001.TIF  # 图片1
│   ├── USD1107373-20251230-D00002.TIF  # 图片2
│   └── ...                        # 可能有多张图片
│
├── USD1107374/
│   ├── USD1107374-20251230.XML
│   └── ...
│
└── ...
```

**关键点**：
- **一个专利 = 一个目录 = 一个 XML + 多张图片**
- 图片数量不固定，少则 1-2 张，多则 10+ 张
- XML 包含专利的所有元数据

### 10.2 XML 元数据解析

`scripts/design_patent_parser.py` 负责解析 XML，提取以下信息：

```python
@dataclass
class DesignPatent:
    """外观专利数据结构"""

    # 核心标识
    patent_id: str       # 专利号，如 "D1107392"
    kind: str            # 文献类型，通常是 "S1"

    # 描述信息
    title: str           # 设计名称，如 "Watch band"
    loc_class: str       # LOC分类号，如 "10-02"（计时仪器类）
    claim_text: str      # 权利要求文本

    # 日期信息
    pub_date: int        # 公开日期 YYYYMMDD
    filing_date: int     # 申请日期 YYYYMMDD
    grant_term: int      # 授权期限（年），通常15年

    # 当事人
    applicant_name: str      # 申请人（公司或个人）
    applicant_country: str   # 申请人国家
    inventor_names: str      # 发明人列表（逗号分隔）
    assignee_name: str       # 受让人（通常是公司）

    # 图片信息
    images: list[str]    # 图片文件名列表
    image_count: int     # 图片数量
```

**XML 解析核心代码**：

```python
def parse_design_patent_xml(xml_path: str) -> DesignPatent:
    """解析 USPTO 外观专利 XML"""

    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 根元素必须是 us-patent-grant
    if root.tag != 'us-patent-grant':
        return None

    # 获取书目数据节点
    biblio = root.find('us-bibliographic-data-grant')

    # 解析专利号
    pub_ref = biblio.find('publication-reference/document-id')
    patent_id = pub_ref.find('doc-number').text  # 如 "D1107392"

    # 解析标题
    title = biblio.find('invention-title').text

    # 解析 LOC 分类（洛迦诺分类）
    loc = biblio.find('classification-locarno')
    loc_class = loc.find('main-classification').text  # 如 "10-02"

    # 解析图片列表
    images = []
    for img in root.findall('.//drawings/figure/img'):
        file_name = img.get('file')  # 如 "USD1107392-20251230-D00001.TIF"
        images.append(file_name)

    # ...更多字段解析
```

### 10.3 数据切分策略

**核心问题**：一个专利有多张图片，如何存入 Milvus？

**当前策略：每张图片一条记录**

```
专利 USD1107373（3张图片）
    │
    ├─→ 记录1: D00001.TIF → 向量1 + 元数据
    ├─→ 记录2: D00002.TIF → 向量2 + 元数据
    └─→ 记录3: D00003.TIF → 向量3 + 元数据

# Milvus 中：
| id | patent_id   | image_index | file_name           | embedding      | title      | ... |
|----|-------------|-------------|---------------------|----------------|------------|-----|
| 1  | USD1107373  | 0           | D00001.TIF          | [0.1, 0.2,...] | Watch band | ... |
| 2  | USD1107373  | 1           | D00002.TIF          | [0.3, 0.1,...] | Watch band | ... |
| 3  | USD1107373  | 2           | D00003.TIF          | [0.2, 0.4,...] | Watch band | ... |
```

**为什么这样设计？**

| 方案 | 优点 | 缺点 |
|------|------|------|
| 每张图片一条记录 | 可以匹配到具体哪张图最相似 | 同一专利可能返回多次 |
| 每个专利一条记录（多向量平均） | 结果简洁 | 丢失细节，无法知道哪张图匹配 |
| 每个专利一条记录（选代表图） | 结果简洁 | 可能选错，漏掉相似图 |

**当前方案的优势**：
- 搜索精度高，能找到最相似的具体图片
- 后处理时可以按 `patent_id` 归组（`group_by_patent` 函数）

### 10.4 完整导入流程

`scripts/import_design_patents_full.py` 的工作流程：

```
┌─────────────────────────────────────────────────────────────────────┐
│                  外观专利导入流程                                     │
└─────────────────────────────────────────────────────────────────────┘

1. 扫描数据目录
   │  scan_design_patents("data/")
   │  查找所有 USD* 子目录
   │
   ▼
2. 解析 XML 元数据
   │  for patent_dir in data_path.glob('USD*'):
   │      xml_path = patent_dir.glob('*.XML')[0]
   │      patent = parse_design_patent_xml(xml_path)
   │
   │  得到 DesignPatent 对象列表
   │
   ▼
3. 遍历每个专利的每张图片
   │
   │  for patent in patents:
   │      for img_idx, img_file in enumerate(patent.images):
   │          │
   │          ├─→ 3.1 上传图片到 MinIO
   │          │       object_name = f"design_patents/{patent_id}/{img_file}"
   │          │       minio_url = minio_service.upload_file(local_path, object_name)
   │          │
   │          ├─→ 3.2 DINOv2 向量化
   │          │       embedding = dinov2_base_extractor.extract_single(local_path)
   │          │
   │          └─→ 3.3 准备 Milvus 数据
   │                  {
   │                      "patent_id": patent.patent_id,
   │                      "image_index": img_idx,        # 图片序号
   │                      "file_name": img_file,
   │                      "file_path": minio_url,
   │                      "embedding": embedding,        # 768维向量
   │                      "title": patent.title,
   │                      "loc_class": patent.loc_class,
   │                      "applicant_name": patent.applicant_name,
   │                      # ...其他元数据
   │                  }
   │
   ▼
4. 批量插入 Milvus
   │  每 32 条记录插入一次
   │  collection.insert(entities)
   │
   ▼
5. 完成统计
   │  成功: X 张图片
   │  失败: Y 张图片
   │  Collection 总记录: Z
```

### 10.5 Collection Schema（外观专利完整版）

```python
# design_patents_full Collection 的字段定义

fields = [
    # 主键
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),

    # 核心字段（搜索必需）
    FieldSchema(name="patent_id", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="image_index", dtype=DataType.INT16),      # 图片序号
    FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=512),  # MinIO URL
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),    # Base模型

    # 元数据字段（用于展示和过滤）
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
    FieldSchema(name="loc_class", dtype=DataType.VARCHAR, max_length=20),
    FieldSchema(name="loc_edition", dtype=DataType.VARCHAR, max_length=10),
    FieldSchema(name="pub_date", dtype=DataType.INT64),         # 公开日期
    FieldSchema(name="filing_date", dtype=DataType.INT64),      # 申请日期
    FieldSchema(name="grant_term", dtype=DataType.INT16),       # 授权期限
    FieldSchema(name="applicant_name", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="applicant_country", dtype=DataType.VARCHAR, max_length=10),
    FieldSchema(name="inventor_names", dtype=DataType.VARCHAR, max_length=500),
    FieldSchema(name="assignee_name", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="claim_text", dtype=DataType.VARCHAR, max_length=500),
    FieldSchema(name="image_count", dtype=DataType.INT16),      # 该专利图片总数
    FieldSchema(name="created_at", dtype=DataType.INT64),
]
```

### 10.6 数据量建议与扩展策略

#### 当前数据量评估

| 指标 | 典型值 | 说明 |
|------|--------|------|
| 每个专利的图片数 | 3-10 张 | 平均约 5 张 |
| 每张图片的向量大小 | 768 × 4 bytes = 3KB | Base 模型 |
| 每条记录的元数据 | ~1KB | VARCHAR 字段 |
| 每条记录总大小 | ~4KB | 向量 + 元数据 |

#### 扩展建议

**1. 数据量对索引的影响**

| 数据量 | 索引配置 | 搜索时间 |
|--------|----------|----------|
| < 10 万 | nlist=128, nprobe=16 | <10ms |
| 10-100 万 | nlist=256, nprobe=32 | 10-50ms |
| 100-1000 万 | nlist=1024, nprobe=64 | 50-100ms |
| > 1000 万 | 考虑 HNSW 索引或分片 | - |

**2. 是否应该灌更多数据？**

```
建议：是的，数据越多效果越好！

原因：
1. 向量检索是"找最相似的"，数据越多，越可能找到真正相似的
2. IVF 索引在数据量大时效果更好（聚类更准确）
3. 对于专利检索，覆盖面广很重要

但要注意：
1. 确保 Milvus 服务器有足够内存（约 4GB/百万条记录）
2. 导入时间会增加（约 1-2 秒/张图片，GPU）
3. 可以分批导入，支持断点续传
```

**3. 如何获取更多 USPTO 数据**

USPTO 提供免费的批量下载：
- 官网：https://bulkdata.uspto.gov/
- 外观专利：`Patent Grant Full Text Data (Design Patents)`
- 格式：XML + 图片打包

**4. 扩展到其他类型数据**

当前系统架构支持多种数据源：

```python
# 可以创建不同的 Collection 存储不同数据

COLLECTIONS = {
    "design_patents_full": "USPTO 外观专利",
    "patent_images_base": "普通专利图片",
    "ruiguan_images": "睿观产品图片",
    "product_catalog": "产品目录图片",
}

# 搜索时可以指定在哪个 Collection 中搜索
# 或者同时搜索多个 Collection 并合并结果
```

### 10.7 运行导入脚本

```bash
# 1. 确保 Milvus 和 MinIO 服务已启动

# 2. 准备数据目录
# 将 USPTO 数据解压到项目根目录，形成 USD*/XXX.XML 结构

# 3. 运行导入脚本
cd scripts
python import_design_patents_full.py

# 输出示例：
# ============================================================
# 外观专利完整导入 (MinIO + Milvus)
# ============================================================
# [DATA] 数据目录: D:\dev\develop\dinov2_milvus_Test
# [MILVUS] Connecting to 192.168.1.174:31278...
# [MILVUS] Connected
# [SCAN] 扫描外观专利...
# [SCAN] 共 150 个专利
# [IMPORT] 开始导入: 150 专利, 723 图片
# [1/150] D1107373: Watch band...
#   进度: 0.7% | 成功: 3 | 失败: 0
# ...
# ============================================================
# 导入完成!
# 成功: 720
# 失败: 3
# Collection 总记录: 720
# ============================================================
```

---

## 11. 设计模式与最佳实践

### 11.1 单例模式 (Singleton)

```python
class MilvusService:
    _instance = None  # 类变量，存储唯一实例

    def __new__(cls):
        """
        __new__ 在 __init__ 之前调用
        通过控制实例创建来实现单例
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

# 使用：
service1 = MilvusService()
service2 = MilvusService()
print(service1 is service2)  # True，同一个实例
```

**为什么用单例？**
- Milvus 连接不需要多个
- DINOv2 模型占用大量 GPU 内存
- 避免重复创建浪费资源

### 11.2 延迟初始化 (Lazy Initialization)

```python
class DINOv2Extractor:
    def __init__(self):
        self._initialized = False
        self.model = None

    def _ensure_initialized(self):
        """首次使用时才初始化"""
        if self._initialized:
            return

        # 耗时操作：加载模型
        self.model = AutoModel.from_pretrained(...)
        self._initialized = True

    def extract(self, image):
        self._ensure_initialized()  # 确保已初始化
        return self.model(image)
```

**为什么用延迟初始化？**
- 应用启动更快
- 内存按需使用
- 某些服务可能根本不用到

### 11.3 分层架构

```
┌─────────────────┐
│     API 层      │  处理 HTTP 请求/响应
├─────────────────┤
│   Service 层    │  业务逻辑
├─────────────────┤
│   Data 层       │  数据访问 (Milvus/MinIO)
└─────────────────┘
```

**好处**：
- 职责分离，代码清晰
- 易于测试
- 可以独立修改某一层

### 11.4 后台线程初始化

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 在后台线程中初始化，不阻塞 API 服务
    init_thread = threading.Thread(target=init_milvus_in_background, daemon=True)
    init_thread.start()

    yield  # API 服务立即可用
```

**为什么用后台线程？**
- Milvus 连接可能需要几秒
- 用户可以立即看到 API 文档
- 提升用户体验

---

## 12. 生产环境注意事项

### 12.1 安全配置

```python
# 当前配置（开发环境）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源 ⚠️ 危险
)

# 生产环境应该
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://your-domain.com",
        "https://admin.your-domain.com",
    ],
)
```

### 12.2 配置外部化

```python
# 当前配置（硬编码）
MILVUS_HOST = "192.168.1.174"

# 生产环境应该（环境变量）
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")

# 或使用 .env 文件 + python-dotenv
```

### 12.3 错误处理

```python
# 当前方式
try:
    results = milvus_service.search(...)
except Exception as e:
    return {"code": 1, "message": str(e)}

# 生产环境应该
from fastapi import HTTPException

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    # 记录日志
    logger.error(f"Error: {exc}", exc_info=True)

    # 返回友好错误信息（不暴露内部细节）
    return JSONResponse(
        status_code=500,
        content={"code": 500, "message": "Internal server error"}
    )
```

### 12.4 日志配置

```python
# 当前方式
print(f"[MILVUS] Connected")

# 生产环境应该
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info("Milvus connected")
```

### 12.5 性能优化建议

| 问题 | 当前状态 | 优化建议 |
|------|----------|----------|
| 模型推理 | 单次处理 | 批量推理 |
| 缩略图 | 实时生成 | 预生成缓存 |
| 图片存储 | 本地 + MinIO | CDN 加速 |
| 数据库连接 | 单连接 | 连接池 |
| API 响应 | 同步 | 异步 + 缓存 |

### 12.6 监控与健康检查

```python
# 当前健康检查
@app.get("/health")
async def health():
    return {"status": "ok"}

# 生产环境应该
@app.get("/health")
async def health():
    status = {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "components": {}
    }

    # 检查 Milvus
    try:
        milvus_service.get_stats()
        status["components"]["milvus"] = "ok"
    except:
        status["components"]["milvus"] = "error"
        status["status"] = "degraded"

    # 检查 MinIO
    try:
        minio_service.client.list_buckets()
        status["components"]["minio"] = "ok"
    except:
        status["components"]["minio"] = "error"
        status["status"] = "degraded"

    # 检查 GPU
    if torch.cuda.is_available():
        status["components"]["gpu"] = "ok"
    else:
        status["components"]["gpu"] = "not_available"

    return status
```

---

## 附录 A：常见问题

### Q1: 为什么搜索结果不准确？

可能原因：
1. **min_score 设置太高**：降低到 0.3-0.4
2. **数据量太少**：Milvus 索引需要一定数据量才能发挥效果
3. **图片预处理不一致**：确保导入和搜索使用相同的预处理

### Q2: GPU 内存不足怎么办？

解决方案：
1. 减小 BATCH_SIZE（从 32 降到 16 或 8）
2. 使用更小的模型（dinov2-small）
3. 清理 GPU 缓存：`torch.cuda.empty_cache()`

### Q3: Milvus 连接失败怎么办？

检查步骤：
1. 确认 Milvus 服务运行中
2. 检查网络连通性：`telnet 192.168.1.174 31278`
3. 检查防火墙设置

---

## 附录 B：扩展阅读

- [FastAPI 官方文档](https://fastapi.tiangolo.com/)
- [Milvus 官方文档](https://milvus.io/docs)
- [DINOv2 论文](https://arxiv.org/abs/2304.07193)
- [Pillow 文档](https://pillow.readthedocs.io/)
- [PyTorch 文档](https://pytorch.org/docs/)

---

*文档生成时间：2026-01-13*
*如有问题，请联系开发团队*
