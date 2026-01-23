# DINOv2 图像检索系统开发文档

## 1. 项目概述

基于 DINOv2 模型和 Milvus 向量数据库构建的图像相似度检索系统。

**技术栈：**
- 后端：FastAPI + Python
- 向量数据库：Milvus (192.168.1.174:31902)
- 特征提取：DINOv2 (facebook/dinov2-small)
- 前端：Vue 3 + Vite

**数据规模：** 6387 张 TIF 图片

---

## 2. 设计决策（已确认）

| 配置项 | 决策 |
|-------|------|
| 搜索返回数量 | Top-10 |
| 相似度阈值 | 需要，建议 0.5 |
| 缩略图尺寸 | 300x300 |
| 结果排序 | 按相似度降序 |
| 同专利多页 | 归组显示 |
| 批量导入 | 前端触发 |

---

## 3. 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         前端 (Vue 3)                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │ 图片上传  │  │ 图片搜索  │  │ 结果展示  │  │ 性能指标面板     │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │ HTTP API
┌─────────────────────────────────────────────────────────────────┐
│                       后端 (FastAPI)                             │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │ 图片处理服务  │  │ 向量检索服务  │  │ 批量导入服务           │ │
│  └──────────────┘  └──────────────┘  └────────────────────────┘ │
│           │                │                    │                │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │                   DINOv2 特征提取模块                       │ │
│  │           facebook/dinov2-small (384维向量)                 │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────────┐
│                      Milvus 向量数据库                           │
│              Collection: patent_images                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Milvus Collection 设计

### 4.1 Collection Schema

```python
from pymilvus import Collection, FieldSchema, CollectionSchema, DataType

fields = [
    # 主键，自增ID
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),

    # 完整文件名，用于定位原文件
    FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=256),

    # 专利号 (USD1107373)，用于归组显示
    FieldSchema(name="patent_id", dtype=DataType.VARCHAR, max_length=64),

    # 页码 (D00001)，用于排序同专利的多页
    FieldSchema(name="page_num", dtype=DataType.VARCHAR, max_length=16),

    # 文件路径，用于读取原图
    FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=512),

    # DINOv2 特征向量，384维 (dinov2-small)
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),

    # 入库时间戳，便于追踪
    FieldSchema(name="created_at", dtype=DataType.INT64),
]

schema = CollectionSchema(fields, description="Patent image embeddings")
collection = Collection(name="patent_images", schema=schema)
```

### 4.2 字段设计说明

| 字段 | 类型 | 用途 |
|------|------|------|
| `id` | INT64 | 主键，Milvus 自动生成 |
| `file_name` | VARCHAR(256) | 完整文件名，生成缩略图URL、定位文件 |
| `patent_id` | VARCHAR(64) | 专利号，用于搜索结果归组显示 |
| `page_num` | VARCHAR(16) | 页码，同专利内按页码排序 |
| `file_path` | VARCHAR(512) | 完整路径，读取原图时使用 |
| `embedding` | FLOAT_VECTOR(384) | DINOv2-small CLS token 输出向量 |
| `created_at` | INT64 | Unix 时间戳，追踪入库时间 |

### 4.3 为什么是 384 维？

```
DINOv2 模型系列：
┌─────────────────┬──────────┬─────────────┐
│ 模型            │ 向量维度  │ 参数量      │
├─────────────────┼──────────┼─────────────┤
│ dinov2-small    │ 384      │ 22M         │  ← 我们使用这个
│ dinov2-base     │ 768      │ 86M         │
│ dinov2-large    │ 1024     │ 300M        │
│ dinov2-giant    │ 1536     │ 1.1B        │
└─────────────────┴──────────┴─────────────┘

选择 small 的原因：
├── 6387 张图片规模较小，不需要超大模型
├── 推理速度快，适合实时搜索
├── 显存占用小，批量处理更高效
└── 精度对于图像相似度检索已足够
```

---

## 5. 索引设计详解

### 5.1 索引配置

```python
index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "COSINE",
    "params": {"nlist": 128}
}
collection.create_index("embedding", index_params)
```

### 5.2 为什么选 IVF_FLAT？

**Milvus 常用索引对比：**

| 索引类型 | 适用规模 | 精度 | 速度 | 内存占用 | 说明 |
|---------|---------|------|------|---------|------|
| FLAT | <1万 | 100% | 慢 | 低 | 暴力搜索，无压缩 |
| **IVF_FLAT** | 1万-100万 | 99%+ | 快 | 中 | 聚类+暴力，精度高 |
| IVF_PQ | 100万+ | 95%+ | 很快 | 低 | 聚类+量化压缩 |
| HNSW | 任意 | 99%+ | 很快 | 高 | 图结构，内存换速度 |

**我们的场景：6387 张图片**

```
IVF_FLAT 最合适：
├── 数据量适中 (6387)，不需要压缩
├── 保持高精度（接近暴力搜索的 99%+）
├── 搜索速度足够快（毫秒级）
└── 内存占用可接受
```

### 5.3 IVF_FLAT 工作原理

```
                    建索引阶段

原始数据 (6387个向量)
        │
        ▼ K-means 聚类
┌───────────────────────────────────────────────┐
│  簇1        簇2        簇3    ...    簇128    │  (nlist=128 个簇)
│  中心点     中心点     中心点         中心点   │
│    │          │          │             │      │
│ ┌──┴──┐   ┌──┴──┐   ┌──┴──┐      ┌──┴──┐   │
│ │向量 │   │向量 │   │向量 │      │向量 │   │
│ │向量 │   │向量 │   │向量 │      │向量 │   │
│ │ ... │   │ ... │   │ ... │      │ ... │   │
│ └─────┘   └─────┘   └─────┘      └─────┘   │
│  ~50个     ~50个     ~50个        ~50个     │
└───────────────────────────────────────────────┘


                    搜索阶段

查询向量 Q
    │
    ▼
1. 计算 Q 与所有 128 个簇中心的距离
    │
    ▼
2. 选择最近的 nprobe=16 个簇
    │
    ▼
3. 只在这 16 个簇内暴力搜索 (~800个向量)
    │
    ▼
4. 返回 Top-10 最相似结果

效率提升：只搜索 800/6387 ≈ 12.5% 的数据
```

### 5.4 nlist = 128 的选择

```
经验公式：nlist ≈ 4 × √n

你的数据：n = 6387
计算：4 × √6387 ≈ 4 × 80 ≈ 320

但实际选择 128 的原因：
├── 数据量较小时，nlist 过大反而慢（簇太小，聚类效果差）
├── 128 是 2 的幂次，计算效率高
├── 每簇约 50 个向量，粒度适中
└── 保守值，平衡精度和速度

如果数据量增长到 10万+，可调整为 512 或 1024
```

### 5.5 为什么用 COSINE 而不是 L2？

```
┌─────────────────────────────────────────────────────────────┐
│                    COSINE vs L2 (欧氏距离)                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  L2 距离：考虑向量的"长度"和"方向"                           │
│                                                             │
│      A ────────────→                                        │
│                      } L2 距离 = 向量端点间的直线距离        │
│      B ──────→                                              │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  COSINE 相似度：只考虑向量的"方向"，忽略"长度"               │
│                                                             │
│      A ────────────→                                        │
│                       θ = 夹角                              │
│      B ──────→                                              │
│                                                             │
│  cos(θ) = 1  → 完全相同方向                                 │
│  cos(θ) = 0  → 正交（无关）                                 │
│  cos(θ) = -1 → 完全相反方向                                 │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  DINOv2 特征的特点：                                        │
│  ├── 输出是高维语义特征                                     │
│  ├── 特征的"方向"代表语义信息                               │
│  ├── 特征的"长度"可能受图像亮度等因素影响                   │
│  └── COSINE 只比较语义方向，更稳定                          │
│                                                             │
│  结论：COSINE 更适合语义相似度比较                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 5.6 搜索参数配置

```python
search_params = {
    "metric_type": "COSINE",
    "params": {
        "nprobe": 16  # 搜索时检查的簇数量
    }
}
```

**nprobe 参数详解：**

| nprobe | 检查的簇 | 检查的向量(约) | 召回率 | 速度 |
|--------|---------|---------------|-------|------|
| 1 | 1/128 | ~50 | ~70% | 最快 |
| 8 | 8/128 | ~400 | ~95% | 快 |
| **16** | 16/128 | ~800 | **~99%** | 适中 |
| 32 | 32/128 | ~1600 | ~99.9% | 较慢 |
| 128 | 128/128 | 6387 | 100% | 等于暴力搜索 |

**选择 nprobe=16 的原因：**
- 检查 12.5% 的数据即可达到 99% 召回率
- 搜索速度在毫秒级
- 对于 Top-10 结果足够准确

---

## 6. 相似度过滤

### 6.1 过滤策略

```python
# COSINE 相似度范围：[-1, 1]
# Milvus 返回的是归一化后的 [0, 1] 或 距离值
# 设置最低阈值过滤无关结果

MIN_SIMILARITY_SCORE = 0.5  # 可配置
```

### 6.2 实现方式

```python
# Milvus 向量搜索时无法直接过滤分数
# 需要在应用层过滤

results = collection.search(
    data=[query_vector],
    anns_field="embedding",
    param=search_params,
    limit=20,  # 多取一些，过滤后可能不足 10 个
    output_fields=["file_name", "patent_id", "page_num", "file_path"]
)

# 应用层过滤
MIN_SCORE = 0.5
filtered_results = []
for hit in results[0]:
    if hit.score >= MIN_SCORE:  # COSINE 相似度越大越相似
        filtered_results.append({
            "id": hit.id,
            "score": hit.score,
            "file_name": hit.entity.get("file_name"),
            "patent_id": hit.entity.get("patent_id"),
            "page_num": hit.entity.get("page_num"),
        })

# 取 Top-10
final_results = filtered_results[:10]
```

### 6.3 相似度分数解释

```
分数范围 (COSINE)：
├── 0.9 - 1.0  : 非常相似（可能是同一图片或微小变化）
├── 0.7 - 0.9  : 高度相似（相似的图案/结构）
├── 0.5 - 0.7  : 中等相似（有一定相关性）
├── 0.3 - 0.5  : 低相似度（弱相关）
└── 0.0 - 0.3  : 几乎不相关

建议阈值：0.5（过滤掉明显不相关的结果）
```

---

## 7. 图片批量处理方案

### 7.1 批量导入策略

**问题：** 6387 张 TIF 图片，单张处理太慢

**方案：** 分批处理 + 进度追踪 + 前端触发

```python
# 推荐配置
BATCH_SIZE = 32          # 每批处理数量（根据显存调整）
MILVUS_INSERT_BATCH = 100  # 每批插入 Milvus 数量
```

### 7.2 处理流程

```
1. 前端触发 POST /api/batch/start
2. 后端启动异步任务
3. 扫描目录获取所有图片路径
4. 检查 Milvus 中已存在的文件（断点续传）
5. 分批加载图片 → TIF 转 RGB
6. 批量提取 DINOv2 特征
7. 批量插入 Milvus
8. 同时生成缩略图
9. 更新进度（前端轮询 GET /api/batch/status）
```

### 7.3 特征提取代码

```python
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image

class DINOv2Extractor:
    def __init__(self, model_name='facebook/dinov2-small', device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def extract_single(self, image_path: str) -> list[float]:
        """提取单张图片特征，返回 384 维向量"""
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # 使用 CLS token 作为图像特征 (第一个 token)
            features = outputs.last_hidden_state[:, 0, :]
            # L2 归一化（配合 COSINE 相似度）
            features = torch.nn.functional.normalize(features, p=2, dim=1)

        return features.squeeze().cpu().tolist()

    def extract_batch(self, image_paths: list[str]) -> list[list[float]]:
        """批量提取特征"""
        images = [Image.open(p).convert('RGB') for p in image_paths]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            features = outputs.last_hidden_state[:, 0, :]
            features = torch.nn.functional.normalize(features, p=2, dim=1)

        return features.cpu().tolist()
```

### 7.4 进度追踪

```json
// import_progress.json
{
  "status": "running",
  "total": 6387,
  "processed": 3200,
  "failed": ["USD1107373-20251230-D00001.TIF"],
  "last_batch": "2024-01-07T10:30:00",
  "start_time": "2024-01-07T10:00:00",
  "avg_speed_per_sec": 17.8
}
```

---

## 8. API 接口设计

### 8.1 接口列表

| 接口 | 方法 | 描述 |
|------|------|------|
| `/api/search` | POST | 以图搜图 |
| `/api/batch/start` | POST | 启动批量导入 |
| `/api/batch/status` | GET | 获取导入进度 |
| `/api/collection/stats` | GET | 获取 Collection 统计 |
| `/api/image/{file_name}` | GET | 获取缩略图 |
| `/api/image/full/{file_name}` | GET | 获取原图 |

### 8.2 搜索接口详细设计

**请求：**
```
POST /api/search
Content-Type: multipart/form-data

file: <image_file>
top_k: 10 (可选，默认10)
min_score: 0.5 (可选，默认0.5)
```

**响应：**
```json
{
  "code": 0,
  "message": "success",
  "data": {
    "results": [
      {
        "patent_id": "USD1107373",
        "pages": [
          {
            "id": 12345,
            "file_name": "USD1107373-20251230-D00001.TIF",
            "page_num": "D00001",
            "score": 0.9523,
            "thumbnail_url": "/api/image/USD1107373-20251230-D00001.TIF"
          },
          {
            "id": 12346,
            "file_name": "USD1107373-20251230-D00002.TIF",
            "page_num": "D00002",
            "score": 0.8912,
            "thumbnail_url": "/api/image/USD1107373-20251230-D00002.TIF"
          }
        ],
        "max_score": 0.9523
      }
    ],
    "timing": {
      "feature_extraction_ms": 45,
      "milvus_search_ms": 12,
      "post_process_ms": 5,
      "total_ms": 62
    },
    "query_info": {
      "top_k": 10,
      "min_score": 0.5,
      "total_matched": 8
    }
  }
}
```

**归组逻辑：**
- 搜索结果按 `patent_id` 分组
- 每组内按 `page_num` 排序
- 组间按 `max_score`（该专利最高分）降序排列

### 8.3 批量导入进度接口

**响应：**
```json
{
  "code": 0,
  "data": {
    "status": "running",
    "total": 6387,
    "processed": 3200,
    "failed": 2,
    "progress_percent": 50.1,
    "estimated_remaining_sec": 180,
    "current_batch": 32,
    "avg_speed": "17.8 images/sec",
    "elapsed_sec": 180
  }
}
```

---

## 9. 前端设计方案

### 9.1 页面布局

```
┌────────────────────────────────────────────────────────────────┐
│  [Logo] DINOv2 图像检索系统            [数据库状态: 6387张]     │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │                      上传区域                            │  │
│  │         拖拽图片到此处，或点击上传                        │  │
│  │              支持 TIF / JPG / PNG                        │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                │
│  ┌──────────────┐  ┌───────────────────────────────────────┐  │
│  │   查询图片    │  │            性能指标面板               │  │
│  │              │  │  ┌─────────┐ ┌─────────┐ ┌─────────┐  │  │
│  │   [预览图]   │  │  │特征提取 │ │向量检索 │ │  总耗时  │  │  │
│  │              │  │  │  45ms   │ │  12ms   │ │  62ms   │  │  │
│  │              │  │  └─────────┘ └─────────┘ └─────────┘  │  │
│  └──────────────┘  └───────────────────────────────────────┘  │
│                                                                │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  搜索结果 (按专利归组)                    共 8 个结果    │  │
│  │                                                         │  │
│  │  ┌─────────────────────────────────────────────────┐   │  │
│  │  │ USD1107373  最高相似度: 95.2%                    │   │  │
│  │  │ ┌────────┐ ┌────────┐ ┌────────┐               │   │  │
│  │  │ │ D00001 │ │ D00002 │ │ D00003 │               │   │  │
│  │  │ │ 95.2%  │ │ 89.1%  │ │ 85.3%  │               │   │  │
│  │  │ └────────┘ └────────┘ └────────┘               │   │  │
│  │  └─────────────────────────────────────────────────┘   │  │
│  │                                                         │  │
│  │  ┌─────────────────────────────────────────────────┐   │  │
│  │  │ USD1107374  最高相似度: 82.7%                    │   │  │
│  │  │ ┌────────┐ ┌────────┐                          │   │  │
│  │  │ │ D00001 │ │ D00002 │                          │   │  │
│  │  │ │ 82.7%  │ │ 78.3%  │                          │   │  │
│  │  │ └────────┘ └────────┘                          │   │  │
│  │  └─────────────────────────────────────────────────┘   │  │
│  │                                                         │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 9.2 核心组件

```
src/
├── components/
│   ├── ImageUploader.vue      # 图片上传组件（拖拽+点击）
│   ├── SearchResults.vue      # 搜索结果容器（归组展示）
│   ├── PatentGroup.vue        # 单个专利分组
│   ├── ResultCard.vue         # 单个结果卡片
│   ├── TimingPanel.vue        # 性能指标面板
│   ├── DatabaseStatus.vue     # 数据库状态指示器
│   └── BatchImportPanel.vue   # 批量导入进度面板
├── views/
│   ├── SearchView.vue         # 主搜索页面
│   └── AdminView.vue          # 管理页面（批量导入）
├── api/
│   └── index.js               # API 调用封装
└── App.vue
```

### 9.3 性能指标展示

```vue
<template>
  <div class="timing-panel">
    <div class="timing-item">
      <div class="timing-value">{{ timing.feature_extraction_ms }}ms</div>
      <div class="timing-label">特征提取</div>
      <div class="timing-bar" :style="{ width: featurePercent + '%' }"></div>
    </div>
    <div class="timing-item">
      <div class="timing-value">{{ timing.milvus_search_ms }}ms</div>
      <div class="timing-label">向量检索</div>
      <div class="timing-bar" :style="{ width: searchPercent + '%' }"></div>
    </div>
    <div class="timing-item total">
      <div class="timing-value">{{ timing.total_ms }}ms</div>
      <div class="timing-label">总耗时</div>
    </div>
  </div>
</template>
```

### 9.4 专利分组卡片

```vue
<template>
  <div class="patent-group">
    <div class="group-header">
      <span class="patent-id">{{ patentId }}</span>
      <span class="max-score">最高相似度: {{ (maxScore * 100).toFixed(1) }}%</span>
    </div>
    <div class="pages-grid">
      <div v-for="page in pages" :key="page.id" class="page-card" @click="showDetail(page)">
        <img :src="page.thumbnail_url" :alt="page.file_name" loading="lazy" />
        <div class="page-info">
          <span class="page-num">{{ page.page_num }}</span>
          <span class="score">{{ (page.score * 100).toFixed(1) }}%</span>
        </div>
        <div class="score-bar" :style="{ width: page.score * 100 + '%' }"></div>
      </div>
    </div>
  </div>
</template>
```

---

## 10. 图片服务方案

### 10.1 问题
TIF 图片浏览器不能直接显示

### 10.2 解决方案：预生成缩略图

```
testImage/
├── USD1107373-20251230-D00000.TIF      # 原图
├── USD1107373-20251230-D00001.TIF
└── thumbnails/                          # 缩略图目录
    ├── USD1107373-20251230-D00000.jpg  # 300x300 JPEG
    └── USD1107373-20251230-D00001.jpg
```

### 10.3 缩略图生成

```python
from PIL import Image
import os

def generate_thumbnail(src_path: str, dst_dir: str, size=(300, 300)):
    """生成缩略图"""
    filename = os.path.basename(src_path)
    name, _ = os.path.splitext(filename)
    dst_path = os.path.join(dst_dir, f"{name}.jpg")

    img = Image.open(src_path)
    # TIF 可能是多页的，只取第一页
    if hasattr(img, 'n_frames') and img.n_frames > 1:
        img.seek(0)
    img = img.convert('RGB')
    img.thumbnail(size, Image.Resampling.LANCZOS)
    img.save(dst_path, 'JPEG', quality=85)

    return dst_path
```

### 10.4 API 端点

```python
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# 静态文件服务
app.mount("/thumbnails", StaticFiles(directory="testImage/thumbnails"), name="thumbnails")

@app.get("/api/image/{file_name}")
async def get_thumbnail(file_name: str):
    """获取缩略图"""
    name, _ = os.path.splitext(file_name)
    thumbnail_path = f"testImage/thumbnails/{name}.jpg"
    return FileResponse(thumbnail_path, media_type="image/jpeg")

@app.get("/api/image/full/{file_name}")
async def get_full_image(file_name: str):
    """获取原图（实时转换为 JPEG）"""
    path = f"testImage/{file_name}"
    img = Image.open(path).convert('RGB')
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=95)
    buffer.seek(0)
    return StreamingResponse(buffer, media_type="image/jpeg")
```

---

## 11. 开发步骤

### Phase 1: 基础设施
1. 创建 Milvus Collection 和索引
2. 实现 DINOv2 特征提取模块
3. 实现单张图片入库测试

### Phase 2: 批量导入
4. 实现批量导入 API
5. 添加进度追踪和断点续传
6. 生成缩略图

### Phase 3: 搜索功能
7. 实现搜索 API（含归组逻辑）
8. 添加性能计时
9. 添加相似度过滤

### Phase 4: 前端开发
10. 搭建 Vue 项目，安装依赖
11. 实现图片上传组件
12. 实现搜索结果展示（归组）
13. 实现性能指标面板
14. 实现批量导入管理页面

### Phase 5: 优化
15. 添加图片缓存
16. 优化批量处理性能
17. 添加错误处理和重试机制

---

## 12. 注意事项

### 12.1 TIF 图片处理
```python
# TIF 可能是多页的，只取第一页
from PIL import Image
img = Image.open(path)
if hasattr(img, 'n_frames') and img.n_frames > 1:
    img.seek(0)  # 取第一帧
img = img.convert('RGB')  # 转为 RGB
```

### 12.2 文件名解析
```python
# USD1107373-20251230-D00001.TIF
# 专利号: USD1107373
# 日期: 20251230
# 页码: D00001
import re

def parse_filename(filename: str):
    match = re.match(r'(USD\d+)-(\d+)-(D\d+)\.TIF', filename)
    if match:
        return {
            'patent_id': match.group(1),
            'date': match.group(2),
            'page_num': match.group(3)
        }
    return None
```

### 12.3 GPU 内存管理
```python
# 如果 GPU 内存不足，减小 batch_size 或使用 CPU
import torch
print(f"GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")

# 批处理时及时清理
torch.cuda.empty_cache()
```

### 12.4 L2 归一化的重要性
```python
# 使用 COSINE 相似度时，必须对向量做 L2 归一化
# 否则结果可能不准确

features = torch.nn.functional.normalize(features, p=2, dim=1)

# 归一化后：||features|| = 1
# 此时 COSINE 相似度 = 点积
```
