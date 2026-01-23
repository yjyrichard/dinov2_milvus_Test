"""应用配置"""
import os

# Milvus 配置
MILVUS_HOST = os.getenv("MILVUS_HOST", "192.168.1.174")
MILVUS_PORT = os.getenv("MILVUS_PORT", "31278")
COLLECTION_NAME = "patent_images"

# DINOv2 配置 (Small 模型)
DINOV2_MODEL = "facebook/dinov2-small"
EMBEDDING_DIM = 384

# DINOv2 配置 (Base 模型) - 使用本地模型
DINOV2_BASE_MODEL = "./dinov2-base"
EMBEDDING_DIM_BASE = 768
COLLECTION_NAME_BASE = "patent_images_base"

# 图片目录 批量导入用介个目录
IMAGE_DIR = os.getenv("IMAGE_DIR", "testImage")
THUMBNAIL_DIR = os.path.join(IMAGE_DIR, "thumbnails")

# 所有图片目录（用于图片服务查找）
IMAGE_DIRS = ["testImage", "睿观"]

# 支持的图片格式
SUPPORTED_IMAGE_FORMATS = {".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 批量处理配置
BATCH_SIZE = 32
MILVUS_INSERT_BATCH = 100

# 搜索配置
DEFAULT_TOP_K = 10
DEFAULT_MIN_SCORE = 0.4

# MinIO 配置
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "192.168.1.174:9000")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "admin")
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "trizhi2026")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "trizhi01")
MINIO_SECURE = False  # HTTP

# 睿观 collection 配置 (新名称格式)
COLLECTION_NAME_RUIGUAN = "ruiguan_images_base_20260109"
COLLECTION_NAME_PATENT_NEW = "patent_images_base_20260109"

# 索引配置
INDEX_PARAMS = {
    "index_type": "IVF_FLAT",
    "metric_type": "COSINE",
    "params": {"nlist": 128}
}

SEARCH_PARAMS = {
    "metric_type": "COSINE",
    "params": {"nprobe": 16}
}
