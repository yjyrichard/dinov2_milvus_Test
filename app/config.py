"""应用配置"""
import os

# Milvus 配置
MILVUS_HOST = os.getenv("MILVUS_HOST", "192.168.1.174")
MILVUS_PORT = os.getenv("MILVUS_PORT", "31278")
COLLECTION_NAME = "patent_images"

# DINOv2 配置 (Small 模型)
DINOV2_MODEL = "facebook/dinov2-small"
EMBEDDING_DIM = 384

# DINOv2 配置 (Base 模型)
DINOV2_BASE_MODEL = "facebook/dinov2-base"
EMBEDDING_DIM_BASE = 768
COLLECTION_NAME_BASE = "patent_images_base"

# 图片目录
IMAGE_DIR = os.getenv("IMAGE_DIR", "testImage") # 睿观·ERiC_files
THUMBNAIL_DIR = os.path.join(IMAGE_DIR, "thumbnails")

# 支持的图片格式
SUPPORTED_IMAGE_FORMATS = {".tif", ".tiff", ".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 批量处理配置
BATCH_SIZE = 32
MILVUS_INSERT_BATCH = 100

# 搜索配置
DEFAULT_TOP_K = 10
DEFAULT_MIN_SCORE = 0.4

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
