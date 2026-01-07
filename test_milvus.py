"""单独测试 Milvus 创建 Collection"""
import time
from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    utility
)

MILVUS_HOST = "192.168.1.174"
MILVUS_PORT = "31278"
COLLECTION_NAME = "patent_images"
EMBEDDING_DIM = 384

print("=" * 50)
print("Milvus Collection 创建测试")
print("=" * 50)

# 1. 连接
print(f"\n[1] 连接 Milvus {MILVUS_HOST}:{MILVUS_PORT}...")
start = time.time()
connections.connect(host=MILVUS_HOST, port=MILVUS_PORT)
print(f"    连接成功，耗时: {time.time() - start:.2f}s")

# 2. 检查 collection 是否存在
print(f"\n[2] 检查 collection '{COLLECTION_NAME}' 是否存在...")
start = time.time()
exists = utility.has_collection(COLLECTION_NAME)
print(f"    存在: {exists}，耗时: {time.time() - start:.2f}s")

# 3. 如果存在，先删除
if exists:
    print(f"\n[3] 删除已存在的 collection...")
    start = time.time()
    utility.drop_collection(COLLECTION_NAME)
    print(f"    删除成功，耗时: {time.time() - start:.2f}s")

# 4. 创建 schema
print(f"\n[4] 创建 Schema...")
start = time.time()
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="patent_id", dtype=DataType.VARCHAR, max_length=64),
    FieldSchema(name="page_num", dtype=DataType.VARCHAR, max_length=16),
    FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=512),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
    FieldSchema(name="created_at", dtype=DataType.INT64),
]
schema = CollectionSchema(fields, description="Patent image embeddings")
print(f"    Schema 创建成功，耗时: {time.time() - start:.2f}s")

# 5. 创建 Collection
print(f"\n[5] 创建 Collection '{COLLECTION_NAME}'...")
print(f"    (如果这一步卡住超过10秒，说明 Milvus 服务器有问题)")
start = time.time()
collection = Collection(name=COLLECTION_NAME, schema=schema)
print(f"    Collection 创建成功，耗时: {time.time() - start:.2f}s")

# 6. 创建索引
print(f"\n[6] 创建索引...")
start = time.time()
index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "COSINE",
    "params": {"nlist": 128}
}
collection.create_index(field_name="embedding", index_params=index_params)
print(f"    索引创建成功，耗时: {time.time() - start:.2f}s")

# 7. 加载 Collection
print(f"\n[7] 加载 Collection 到内存...")
start = time.time()
collection.load()
print(f"    加载成功，耗时: {time.time() - start:.2f}s")

# 8. 验证
print(f"\n[8] 验证...")
print(f"    Collection 名称: {collection.name}")
print(f"    实体数量: {collection.num_entities}")
print(f"    有索引: {collection.has_index()}")

print("\n" + "=" * 50)
print("测试完成！Milvus 工作正常")
print("=" * 50)

connections.disconnect("default")
