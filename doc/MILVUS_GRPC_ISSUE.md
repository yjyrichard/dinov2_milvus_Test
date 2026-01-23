# Milvus 使用指南与问题记录

## 目录
- [问题记录：gRPC 消息大小限制](#问题记录grpc-消息大小限制)
- [Milvus API 完整指南](#milvus-api-完整指南)
- [pymilvus 底层架构](#pymilvus-底层架构)

---

# 问题记录：gRPC 消息大小限制

## 问题描述

批量导入数据到 Milvus 时报错：

```
grpc: received message larger than max (72641199 vs. 67108864)
StatusCode.RESOURCE_EXHAUSTED
```

## 原因分析

gRPC 默认消息大小限制为 **64MB** (67108864 bytes)。当批量插入的数据超过此限制时，会触发 `RESOURCE_EXHAUSTED` 错误。

每条记录包含：
- embedding: 768 维 float32 向量 = 768 × 4 = 3072 bytes
- 元数据字段（patent_id, title, file_path 等）≈ 2000 bytes
- 单条记录约 5KB

当 BATCH_SIZE=32 时，因 gRPC 序列化开销，消息可能膨胀到 70MB+。

## 解决方案

### 方案1：减小批次大小（推荐）

```python
INSERT_BATCH_SIZE = 8  # 从 32 减小到 8
```

### 方案2：增大 gRPC 限制

客户端连接时指定：

```python
connections.connect(
    alias="default",
    host="192.168.1.174",
    port="31278",
    _options=[
        ("grpc.max_send_message_length", 128 * 1024 * 1024),
        ("grpc.max_receive_message_length", 128 * 1024 * 1024),
    ]
)
```

---

# Milvus API 完整指南

## 1. 连接管理

### 连接

```python
from pymilvus import connections

# 基本连接
connections.connect(alias="default", host="192.168.1.174", port="31278")

# 带 gRPC 配置的连接
connections.connect(
    alias="default",
    host="192.168.1.174",
    port="31278",
    _options=[
        ("grpc.max_send_message_length", 128 * 1024 * 1024),
        ("grpc.max_receive_message_length", 128 * 1024 * 1024),
    ]
)
```

**底层调用链**：
```
connections.connect()
  → pymilvus/orm/connections.py: Connections.connect()
  → pymilvus/client/grpc_handler.py: GrpcHandler.__init__()
  → grpc.insecure_channel() 或 grpc.secure_channel()
```

### 断开连接

```python
connections.disconnect(alias="default")
```

## 2. Collection 管理

### 创建 Collection

```python
from pymilvus import Collection, FieldSchema, CollectionSchema, DataType

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=500),
]
schema = CollectionSchema(fields, description="My collection")
collection = Collection(name="my_collection", schema=schema)
```

**底层调用链**：
```
Collection(name, schema)
  → pymilvus/orm/collection.py: Collection.__init__()
  → pymilvus/client/grpc_handler.py: GrpcHandler.create_collection()
  → gRPC: CreateCollection RPC
```

### 删除 Collection

```python
from pymilvus import utility

# 方式1：使用 utility
utility.drop_collection("my_collection")

# 方式2：使用 Collection 对象
collection.drop()
```

**底层调用链**：
```
utility.drop_collection(name)
  → pymilvus/orm/utility.py: drop_collection()
  → pymilvus/client/grpc_handler.py: GrpcHandler.drop_collection()
  → gRPC: DropCollection RPC
```

**注意**：删除操作是不可逆的，会删除所有数据和索引！

### 检查 Collection 是否存在

```python
exists = utility.has_collection("my_collection")
```

### 列出所有 Collection

```python
collections = utility.list_collections()
```

### 获取 Collection 信息

```python
collection = Collection("my_collection")

# 记录数量
count = collection.num_entities

# Schema 信息
schema = collection.schema

# 描述信息
desc = collection.describe()
```

## 3. 数据操作

### 插入数据

```python
# 方式1：列表格式
entities = [
    [1, 2, 3],                    # id 列
    [[0.1]*768, [0.2]*768, ...],  # embedding 列
    ["title1", "title2", ...],    # title 列
]
collection.insert(entities)

# 方式2：字典格式
data = [
    {"id": 1, "embedding": [0.1]*768, "title": "title1"},
    {"id": 2, "embedding": [0.2]*768, "title": "title2"},
]
collection.insert(data)
```

**底层调用链**：
```
collection.insert(data)
  → pymilvus/orm/collection.py: Collection.insert()
  → pymilvus/orm/mutation.py: MutationResult
  → pymilvus/client/grpc_handler.py: GrpcHandler.batch_insert()
  → gRPC: Insert RPC
  → 触发 RESOURCE_EXHAUSTED（如果数据过大）
```

### 删除数据

```python
# 按表达式删除
collection.delete(expr="id in [1, 2, 3]")

# 按主键删除
collection.delete(expr="id == 123")

# 按条件删除
collection.delete(expr="title == 'test'")
```

**底层调用链**：
```
collection.delete(expr)
  → pymilvus/orm/collection.py: Collection.delete()
  → pymilvus/client/grpc_handler.py: GrpcHandler.delete()
  → gRPC: Delete RPC
```

### 更新数据

Milvus 不支持直接更新，需要先删除再插入：

```python
# 1. 删除旧数据
collection.delete(expr="id == 123")

# 2. 插入新数据
collection.insert([new_data])
```

## 4. 索引管理

### 创建索引

```python
index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "COSINE",  # 或 "L2", "IP"
    "params": {"nlist": 128}
}
collection.create_index(field_name="embedding", index_params=index_params)
```

**底层调用链**：
```
collection.create_index()
  → pymilvus/orm/collection.py: Collection.create_index()
  → pymilvus/client/grpc_handler.py: GrpcHandler.create_index()
  → gRPC: CreateIndex RPC
```

**常用索引类型**：
| 索引类型 | 适用场景 | 参数 |
|---------|---------|------|
| FLAT | 小数据集，精确搜索 | 无 |
| IVF_FLAT | 中等数据集 | nlist |
| IVF_SQ8 | 大数据集，节省内存 | nlist |
| HNSW | 高召回率 | M, efConstruction |

### 删除索引

```python
collection.drop_index()
```

### 查看索引信息

```python
index = collection.index()
print(index.params)
```

## 5. 加载与释放

### 加载到内存

```python
# 加载整个 Collection
collection.load()

# 加载指定分区
collection.load(partition_names=["partition1"])
```

**底层调用链**：
```
collection.load()
  → pymilvus/orm/collection.py: Collection.load()
  → pymilvus/client/grpc_handler.py: GrpcHandler.load_collection()
  → gRPC: LoadCollection RPC
```

### 释放内存

```python
collection.release()
```

**注意**：搜索前必须先 load()！

## 6. 搜索与查询

### 向量搜索

```python
search_params = {"metric_type": "COSINE", "params": {"nprobe": 16}}

results = collection.search(
    data=[[0.1]*768],           # 查询向量
    anns_field="embedding",     # 向量字段名
    param=search_params,
    limit=10,                   # 返回数量
    output_fields=["title"],    # 返回的标量字段
    expr="title != ''"          # 过滤条件（可选）
)

for hits in results:
    for hit in hits:
        print(f"ID: {hit.id}, Score: {hit.score}, Title: {hit.entity.get('title')}")
```

**底层调用链**：
```
collection.search()
  → pymilvus/orm/collection.py: Collection.search()
  → pymilvus/client/grpc_handler.py: GrpcHandler.search()
  → gRPC: Search RPC
```

### 标量查询

```python
# 查询所有数据
results = collection.query(
    expr="id >= 0",
    output_fields=["id", "title"],
    limit=1000
)

# 条件查询
results = collection.query(
    expr="title like 'patent%'",
    output_fields=["id", "title", "embedding"]
)
```

**底层调用链**：
```
collection.query()
  → pymilvus/orm/collection.py: Collection.query()
  → pymilvus/client/grpc_handler.py: GrpcHandler.query()
  → gRPC: Query RPC
```

## 7. 数据持久化

### Flush

将内存中的数据持久化到磁盘：

```python
collection.flush()
```

**底层调用链**：
```
collection.flush()
  → pymilvus/orm/collection.py: Collection.flush()
  → pymilvus/client/grpc_handler.py: GrpcHandler.flush()
  → gRPC: Flush RPC
```

### Compact

合并小的数据段，优化存储：

```python
collection.compact()
```

---

# pymilvus 底层架构

## 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│                      用户代码                                │
│         collection.insert() / collection.search()           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    ORM 层 (高级 API)                         │
│              pymilvus/orm/collection.py                     │
│              pymilvus/orm/connections.py                    │
│              pymilvus/orm/utility.py                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Client 层 (gRPC 通信)                      │
│              pymilvus/client/grpc_handler.py                │
│              pymilvus/client/stub.py                        │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      gRPC 协议                               │
│                  Protocol Buffers 序列化                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Milvus Server                            │
│    Proxy → QueryNode / DataNode / IndexNode                 │
└─────────────────────────────────────────────────────────────┘
```

## 源码目录结构

```
pymilvus/
├── __init__.py
├── client/
│   ├── grpc_handler.py    # ★ gRPC 通信核心
│   │   ├── batch_insert()     # 插入数据
│   │   ├── search()           # 向量搜索
│   │   ├── query()            # 标量查询
│   │   ├── delete()           # 删除数据
│   │   ├── create_collection()
│   │   ├── drop_collection()
│   │   ├── create_index()
│   │   └── ...
│   ├── stub.py            # gRPC stub 封装
│   └── types.py           # 类型定义
├── orm/
│   ├── collection.py      # ★ Collection 高级 API
│   ├── connections.py     # 连接管理
│   ├── utility.py         # 工具函数
│   ├── mutation.py        # 插入/删除结果
│   └── search.py          # 搜索结果
├── decorators.py          # ★ 错误处理装饰器
└── exceptions.py          # 异常定义
```

## 关键源码位置

查看 pymilvus 安装位置：

```bash
python -c "import pymilvus; print(pymilvus.__file__)"
# Windows: C:\ProgramData\miniconda3\envs\dinov2\Lib\site-packages\pymilvus\
# Linux: /usr/local/lib/python3.x/site-packages/pymilvus/
```

### 关键文件说明

| 文件 | 作用 | 关键方法 |
|-----|------|---------|
| `client/grpc_handler.py` | gRPC 通信核心 | batch_insert(), search(), query() |
| `orm/collection.py` | Collection 高级 API | insert(), search(), delete() |
| `orm/connections.py` | 连接管理 | connect(), disconnect() |
| `decorators.py` | 错误处理 | @error_handler 装饰器 |

### 错误处理流程

```python
# decorators.py 中的错误处理
@error_handler
def batch_insert(self, ...):
    try:
        response = rf.result()  # gRPC 调用
    except grpc.RpcError as e:
        # 这里捕获 RESOURCE_EXHAUSTED 等错误
        raise MilvusException(...)
```

## 官方文档

- Milvus 官方文档: https://milvus.io/docs
- pymilvus API 参考: https://milvus.io/api-reference/pymilvus/v2.4.x/About.md
- gRPC 配置: https://milvus.io/docs/configure_proxy.md
- GitHub 源码: https://github.com/milvus-io/pymilvus

---

# 本项目修改记录

## 修改文件

`scripts/import_design_patents_full.py`

## 修改内容

1. 添加 `INSERT_BATCH_SIZE = 8`（减小批次避免 gRPC 超限）
2. 添加 `get_existing_keys()` 去重函数
3. 在处理循环中添加去重检查

## 使用方法

```bash
# 追加模式（自动跳过已存在记录，不删除 collection）
python scripts/import_design_patents_full.py --append

# 全新导入（会删除现有 collection！）
python scripts/import_design_patents_full.py
```

**注意**：不带 `--append` 参数会删除现有 collection 并重新创建！
![alt text](image.png)