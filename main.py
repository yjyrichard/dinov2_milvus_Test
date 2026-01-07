"""DINOv2 图像检索系统 - FastAPI 主入口"""
import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import search, batch, image
from app.services.milvus_service import milvus_service


def init_milvus_in_background():
    """在后台线程中初始化 Milvus（避免阻塞应用启动）"""
    try:
        print("[MILVUS-INIT] Background initialization starting...")
        milvus_service.connect()
        milvus_service.create_collection()
        milvus_service.create_index()
        milvus_service.load_collection()
        print("[MILVUS-INIT] Background initialization completed")
    except Exception as e:
        print(f"[MILVUS-INIT] Background initialization failed: {e}")
        import traceback
        traceback.print_exc()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    print("=" * 50)
    print("[APP] Starting DINOv2 Image Search API...")
    print("=" * 50)

    # 在后台线程中初始化 Milvus，不阻塞应用启动
    print("[APP] Starting Milvus initialization in background thread...")
    init_thread = threading.Thread(target=init_milvus_in_background, daemon=True)
    init_thread.start()

    print("=" * 50)
    print("[APP] Application started!")
    print("[APP] API docs: http://localhost:8000/docs")
    print("[APP] Health: http://localhost:8000/health")
    print("[APP] Note: Milvus is initializing in background")
    print("=" * 50)

    yield

    # 关闭时清理
    print("[APP] Shutting down...")


app = FastAPI(
    title="DINOv2 Image Search",
    description="基于 DINOv2 的图像相似度检索系统",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(search.router)
app.include_router(batch.router)
app.include_router(image.router)


@app.get("/")
async def root():
    """健康检查"""
    return {
        "message": "DINOv2 Image Search API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok"}
