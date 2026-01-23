# 外观专利 Milvus 导入操作指南

## 一、架构概述

### 1.1 数据流程

```

                          数据导入流程                                  
                                                                         
  1. 扫描数据目录                                                        
     DESIGN/USD*/ 目录下的 XML 和 TIF 文件                          
                                                                         
   2. 解析 XML 元数据                                                     
       patent_id: D1107392                                           
       title: Shoe                                                   
       loc_class: 0204                                               
       pub_date: 20251230                                            
       filing_date: 20240122                                         
       applicant_name: CONSITEX S.A.                                
       images: [USD1107392-20251230-D00001.TIF, ...]                 
                                                                         
   3. 图片向量化                                                          
      TIF图片 → DINOv2 Base → 768维向量                              
                                                                         
   4. 存入 Milvus                                                        
       design_patents_full Collection                                
         embedding (768维向量)                                     
          patent_id, image_index                                    
          元数据字段 (loc_class, title, pub_date...)                 

```

### 1.2 技术选型

| 组件 | 选型 | 说明 |
|------|------|------|
| 向量数据库 | Milvus | 已部署在 192.168.1.174:31278 |
| 图像向量化 | DINOv2 Base | 768维向量，专注几何结构 |
| 索引类型 | IVF_FLAT | 适合百万级数据 |
| 相似度度量 | COSINE | 余弦相似度 |

## 二、Collection 设计

### 2.1  Collection Schema（含元数据）

Collection 名称: `design_patents_full`

```python
fields = [
    # 主键
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),

    # === 核心字段 ===
    FieldSchema(name="patent_id", dtype=DataType.VARCHAR, max_length=64),      # 专利号 D1107392
    FieldSchema(name="image_index", dtype=DataType.INT16),                      # 图片序号 0,1,2...
    FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=256),      # 文件名
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),        # DINOv2向量

    # === 元数据字段（用于过滤和展示）===
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),          # 设计名称
    FieldSchema(name="loc_class", dtype=DataType.VARCHAR, max_length=20),       # LOC分类 0204
    FieldSchema(name="loc_edition", dtype=DataType.VARCHAR, max_length=10),     # LOC版本 15
    FieldSchema(name="pub_date", dtype=DataType.INT64),                         # 公开日期 20251230
    FieldSchema(name="filing_date", dtype=DataType.INT64),                      # 申请日期 20240122
    FieldSchema(name="grant_term", dtype=DataType.INT16),                       # 授权期限 15年
    FieldSchema(name="applicant_name", dtype=DataType.VARCHAR, max_length=256), # 申请人名称
    FieldSchema(name="applicant_country", dtype=DataType.VARCHAR, max_length=10), # 申请人国家 CH/US
    FieldSchema(name="inventor_names", dtype=DataType.VARCHAR, max_length=500), # 发明人（逗号分隔）
    FieldSchema(name="assignee_name", dtype=DataType.VARCHAR, max_length=256),  # 受让人名称
    FieldSchema(name="claim_text", dtype=DataType.VARCHAR, max_length=500),     # 权利要求
    FieldSchema(name="image_count", dtype=DataType.INT16),                      # 总图片数

    # === 系统字段 ===
    FieldSchema(name="created_at", dtype=DataType.INT64),                       # 导入时间戳
]
```

### 2.2 索引配置

https://mp.weixin.qq.com/s/BSPGfBNA2gw2_Wud8JobQA 

```python
INDEX_PARAMS = {
    "index_type": "IVF_FLAT",
    "metric_type": "COSINE",
    "params": {"nlist": 256}  # 百万级数据建议 256-1024
}

SEARCH_PARAMS = {
    "metric_type": "COSINE",
    "params": {"nprobe": 32}  # 搜索时探测的聚类数
}
```

### 

## 三、XML 解析说明

### 3.1 外观专利 XML 结构

```xml
<us-patent-grant>                           <!-- 根元素 -->
  <us-bibliographic-data-grant>             <!-- 书目数据 -->
    <publication-reference>
      <document-id>
        <doc-number>D1107392</doc-number>   <!-- 专利号 -->
        <kind>S1</kind>                      <!-- 文献类型 -->
        <date>20251230</date>                <!-- 公开日期 -->
      </document-id>
    </publication-reference>

    <classification-locarno>
      <main-classification>0204</main-classification>  <!-- LOC分类 -->
    </classification-locarno>

    <invention-title>Shoe</invention-title>  <!-- 设计名称 -->

    <us-term-of-grant>
      <length-of-grant>15</length-of-grant>  <!-- 授权期限 -->
    </us-term-of-grant>

    <us-parties>
      <us-applicants>...</us-applicants>     <!-- 申请人 -->
      <inventors>...</inventors>              <!-- 发明人 -->
    </us-parties>

    <assignees>...</assignees>               <!-- 受让人（可选）-->
  </us-bibliographic-data-grant>

  <drawings>
    <figure>
      <img file="USD1107392-20251230-D00001.TIF"/>  <!-- 图片列表 -->
    </figure>
  </drawings>

  <claims>
    <claim>
      <claim-text>The ornamental design for a shoe as shown and described.</claim-text>
    </claim>
  </claims>
</us-patent-grant>
```

### 3.2 字段提取映射

| 字段 | XML路径 | 示例值 |
|------|---------|--------|
| patent_id | `//publication-reference/document-id/doc-number` | D1107392 |
| kind | `//publication-reference/document-id/kind` | S1 |
| pub_date | `//publication-reference/document-id/date` | 20251230 |
| filing_date | `//application-reference/document-id/date` | 20240122 |
| title | `//invention-title` | Shoe |
| loc_class | `//classification-locarno/main-classification` | 0204 |
| loc_edition | `//classification-locarno/edition` | 15 |
| grant_term | `//us-term-of-grant/length-of-grant` | 15 |
| applicant_name | `//us-applicants/us-applicant/addressbook/orgname` 或 `first-name + last-name` | CONSITEX S.A. |
| applicant_country | `//us-applicants/us-applicant/addressbook/address/country` | CH |
| inventor_names | `//inventors/inventor/addressbook/first-name + last-name` | Alessandro Sartori |
| assignee_name | `//assignees/assignee/addressbook/orgname` | CONSITEX S.A. |
| images | `//drawings/figure/img/@file` | [USD1107392-20251230-D00001.TIF, ...] |
| claim_text | `//claims/claim/claim-text` | The ornamental design for... |

## 四、使用指南

### 4.1 环境要求

```bash
# Python 依赖
pip install pymilvus>=2.3.0
pip install torch transformers pillow
pip install tqdm  # 进度条

# 确保 Milvus 可访问
telnet 192.168.1.174 31278
```

### 4.2 运行导入脚本

```bash
# 进入项目目录
cd D:\work\20251230需求分析\数据\I20251230_r1\I20251230\DESIGN\dinov2Test

# 运行导入脚本
python scripts/import_design_patents.py

# 或者指定参数
python scripts/import_design_patents.py --data-dir "../" --batch-size 50
```

### 4.3 脚本参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data-dir` | `../` | 数据目录（包含 USD* 子目录） |
| `--batch-size` | 32 | 每批处理的图片数 |
| `--skip-existing` | True | 跳过已导入的专利 |
| `--collection` | design_patents_full | Collection 名称 |

### 4.4 检索示例

```python
from pymilvus import Collection, connections

# 连接
connections.connect(host="192.168.1.174", port="31278")
collection = Collection("design_patents_full")
collection.load()

# 假设 query_vector 是待查询图片的 768 维向量
results = collection.search(
    data=[query_vector],
    anns_field="embedding",
    param={"metric_type": "COSINE", "params": {"nprobe": 32}},
    limit=10,
    expr='loc_class == "0204"',  # 可选：按LOC分类过滤
    output_fields=["patent_id", "title", "loc_class", "applicant_name", "pub_date"]
)

for hit in results[0]:
    print(f"专利号: {hit.entity.get('patent_id')}")
    print(f"名称: {hit.entity.get('title')}")
    print(f"相似度: {hit.score:.4f}")
```

## 五、数据量与性能估算

### 5.1 当前数据量（小规模测试）

| 项目 | 数量 | 说明 |
|------|------|------|
| 外观专利数 | 698 | 当前测试数据 |
| 图片总数 | 6381 | 平均每专利 ~9 张图 |
| 单向量大小 | 3KB | 768 × 4 bytes |
| 单条元数据 | ~1KB | 文本字段 |
| **总存储估算** | **~25MB** | 完全可以全存 Milvus |

**结论：当前数据量很小，全部存储在 Milvus 中完全没问题，内存占用极低。**

### 5.2 百万级数据估算

| 项目 | 估算值 | 说明 |
|------|--------|------|
| 专利数 | 1,000,000 | 百万级 |
| 图片数 | 6,000,000 | 平均每专利 6 张 |
| 向量存储 | ~18GB | 6M × 3KB |
| 内存需求 | ~24GB | 索引 + 数据加载 |
| 导入时间 | ~10-20 小时 | 取决于 GPU 性能 |

### 5.3 索引选择建议

| 数据量 | 推荐索引 | nlist | 说明 |
|--------|----------|-------|------|
| < 10万 | FLAT | - | 精确搜索，无损 |
| 10万-100万 | IVF_FLAT | 256 | 平衡精度和速度 |
| 100万-1000万 | IVF_FLAT | 1024 | 或考虑 IVF_PQ |
| > 1000万 | HNSW 或 IVF_PQ | - | 需要更多内存或压缩 |

## 六、文件结构

```
DESIGN/
├── dinov2Test/                          # 现有 DINOv2 服务
│   ├── app/
│   │   ├── config.py                    # 配置文件
│   │   └── services/
│   │       ├── dinov2_base_service.py   # DINOv2 向量化
│   │       └── milvus_base_service.py   # Milvus 服务
│   └── scripts/
│       └── import_design_patents.py     # 新增：完整导入脚本
│
├── scripts/                             # 新增：独立脚本目录
│   ├── design_patent_parser.py          # XML 解析器
│   ├── design_patent_milvus.py          # Milvus 服务（含元数据）
│   └── import_all_design_patents.py     # 批量导入脚本
│
├── USD1107392-20251230/                 # 专利数据目录
│   ├── USD1107392-20251230.XML          # 元数据
│   ├── USD1107392-20251230-D00000.TIF   # 图片
│   ├── USD1107392-20251230-D00001.TIF
│   └── ...
│
├── USD1107745-20251230/
│   └── ...
│
└── 外观专利Milvus导入操作指南.md         # 本文档
```

## 七、平滑扩展到百万数据

### 7.1 设计原则

1. **相同的 Collection Schema**：无需修改
2. **批量导入**：使用分批插入，每批 1000-10000 条
3. **断点续传**：记录已导入的 patent_id，支持中断恢复
4. **索引调整**：数据量增加后可调整 nlist 参数

### 7.2 扩展步骤

```python
# 1. 使用相同的 Collection，无需重建
collection = Collection("design_patents_full")

# 2. 批量导入新数据
for batch in batches:
    collection.insert(batch)
    collection.flush()  # 每批刷新

# 3. 如需调整索引（可选）
collection.release()
collection.drop_index()
collection.create_index(
    field_name="embedding",
    index_params={
        "index_type": "IVF_FLAT",
        "metric_type": "COSINE",
        "params": {"nlist": 1024}  # 增大 nlist
    }
)
collection.load()
```

### 7.3 性能优化建议

1. **GPU 加速**：使用 GPU 进行 DINOv2 推理
2. **并行处理**：多进程处理 XML 解析
3. **MinIO 存储**：图片存储到 MinIO，Milvus 只存向量和元数据
4. **分区**：考虑按日期或 LOC 分类分区

## 八、常见问题

### Q1: 如何验证导入是否成功？

```python
collection = Collection("design_patents_full")
collection.flush()
print(f"总记录数: {collection.num_entities}")

# 随机检索验证
results = collection.query(
    expr='patent_id == "D1107392"',
    output_fields=["patent_id", "title", "loc_class"]
)
print(results)
```

### Q2: 如何删除重复数据？

```python
# 查找重复
results = collection.query(
    expr='patent_id == "D1107392"',
    output_fields=["id", "image_index"]
)

# 删除指定 ID
if len(results) > expected_count:
    ids_to_delete = [r["id"] for r in results[expected_count:]]
    collection.delete(f"id in {ids_to_delete}")
```

### Q3: 如何更新元数据？

Milvus 不支持直接更新，需要删除后重新插入：

```python
# 1. 删除旧记录
collection.delete(f'patent_id == "D1107392"')

# 2. 插入新记录（包含更新后的元数据）
collection.insert(new_data)
```

---

## 九、小规模测试操作步骤

### 9.1 当前数据概况

```
专利数: 698
XML 文件: 698
TIF 文件: 6381
```

### 9.2 完整导入流程

```bash
# 步骤 1: 上传图片到 MinIO
cd D:\work\20251230需求分析\数据\I20251230_r1\I20251230\DESIGN\dinov2Test
python scripts/upload_design_patents_to_minio.py

# 步骤 2: 导入向量和元数据到 Milvus
python scripts/import_design_patents_full.py
```

### 9.3 存储方案

| 存储位置 | 内容 | 说明 |
|----------|------|------|
| MinIO | TIF 原图 | `design_patents/{patent_id}/{filename}` |
| Milvus | 向量 + 元数据 | 包含 MinIO URL，用于前端展示 |

**为什么全存 Milvus？**
- 当前 6381 条记录，约 25MB，内存占用极低
- 元数据字段不多，查询时直接返回，无需二次查询
- 简化架构，减少依赖

## 十、前端图片详情展示

### 10.1 点击图片查看详情

现有 `ImageDetailModal.vue` 组件已支持展示：
- 专利号 (patent_id)
- 页码 (page_num)
- 文件名 (file_name)
- 相似度 (score)

### 10.2 扩展元数据展示

需要在搜索结果中返回更多字段，修改 `output_fields`：

```python
# 后端搜索时返回完整元数据
results = collection.search(
    data=[query_vector],
    anns_field="embedding",
    param=SEARCH_PARAMS,
    limit=top_k,
    output_fields=[
        "patent_id", "file_name", "file_path",
        "title", "loc_class", "pub_date", "filing_date",
        "applicant_name", "applicant_country",
        "inventor_names", "claim_text", "image_count"
    ]
)
```

### 10.3 前端详情弹窗扩展

在 `ImageDetailModal.vue` 中添加更多字段展示：

```vue
<div class="info-item">
  <span class="info-label">设计名称</span>
  <span class="info-value">{{ imageData.title }}</span>
</div>
<div class="info-item">
  <span class="info-label">LOC分类</span>
  <span class="info-value">{{ imageData.loc_class }}</span>
</div>
<div class="info-item">
  <span class="info-label">申请人</span>
  <span class="info-value">{{ imageData.applicant_name }}</span>
</div>
<div class="info-item">
  <span class="info-label">公开日期</span>
  <span class="info-value">{{ formatDate(imageData.pub_date) }}</span>
</div>
<div class="info-item">
  <span class="info-label">权利要求</span>
  <span class="info-value claim">{{ imageData.claim_text }}</span>
</div>
```

## 十一、关于文字 Embedding

### 11.1 外观专利文字特点

外观专利的文字内容非常少，主要是：
- **title**: 设计名称，如 "Shoe"、"Watch"
- **claim_text**: 权利要求，几乎都是固定格式：
  > "The ornamental design for a shoe as shown and described."

### 11.2 是否需要文字 Embedding？

**不需要。** 原因：
1. 文字太少，embedding 意义不大
2. 外观专利核心是图像，搜索靠图像向量
3. 文字直接存原文用于展示即可

### 11.3 如果确实需要文字搜索

可以用 Milvus 的标量过滤：

```python
# 按标题关键词过滤
results = collection.search(
    data=[query_vector],
    anns_field="embedding",
    param=SEARCH_PARAMS,
    limit=10,
    expr='title like "%Shoe%"'  # 标量过滤
)

# 按 LOC 分类过滤
results = collection.search(
    ...,
    expr='loc_class == "0204"'
)
```

## 十二、API 接口设计

### 12.1 搜索接口

```python
@app.post("/api/search/design")
async def search_design_patents(
    file: UploadFile,
    top_k: int = 10,
    loc_class: str = None  # 可选：LOC分类过滤
):
    # 1. 图片向量化
    embedding = dinov2_extractor.extract(file)

    # 2. 构建过滤条件
    expr = f'loc_class == "{loc_class}"' if loc_class else None

    # 3. 搜索
    results = collection.search(
        data=[embedding],
        anns_field="embedding",
        param=SEARCH_PARAMS,
        limit=top_k,
        expr=expr,
        output_fields=[
            "patent_id", "title", "loc_class", "pub_date",
            "applicant_name", "claim_text", "file_path"
        ]
    )

    return {"results": format_results(results)}
```

### 12.2 专利详情接口

```python
@app.get("/api/patent/{patent_id}")
async def get_patent_detail(patent_id: str):
    results = collection.query(
        expr=f'patent_id == "{patent_id}"',
        output_fields=["*"]
    )
    return {"patent": results[0] if results else None}
```

---

**最后更新**: 2026-01-12
