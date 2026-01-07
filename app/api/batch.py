"""批量导入 API"""
import traceback
from fastapi import APIRouter

from app.services.batch_import_service import batch_import_service

router = APIRouter(prefix="/api/batch", tags=["batch"])


@router.post("/start")
async def start_batch_import():
    """启动批量导入"""
    print("[API] POST /api/batch/start called")

    try:
        if batch_import_service.status == "running":
            print("[API] Import already running")
            return {
                "code": 1,
                "message": "Import already running",
                "data": batch_import_service.get_status()
            }

        # 启动导入（在后台线程中运行）
        print("[API] Starting batch import...")
        batch_import_service.start_import()

        status = batch_import_service.get_status()
        print(f"[API] Import started, status: {status}")

        return {
            "code": 0,
            "message": "Import started",
            "data": status
        }
    except Exception as e:
        print(f"[API] start_batch_import error: {e}")
        traceback.print_exc()
        return {
            "code": 1,
            "message": str(e),
            "data": None
        }


@router.get("/status")
async def get_batch_status():
    """获取导入进度"""
    try:
        status = batch_import_service.get_status()
        return {
            "code": 0,
            "data": status
        }
    except Exception as e:
        print(f"[API] get_batch_status error: {e}")
        traceback.print_exc()
        return {
            "code": 1,
            "message": str(e),
            "data": {
                "status": "error",
                "error_message": str(e)
            }
        }


@router.post("/reset")
async def reset_batch_status():
    """重置导入状态（用于错误恢复）"""
    print("[API] POST /api/batch/reset called")

    try:
        if batch_import_service.status == "running":
            return {
                "code": 1,
                "message": "Cannot reset while running"
            }

        batch_import_service.status = "idle"
        batch_import_service.processed = 0
        batch_import_service.failed = []
        batch_import_service.error_message = None

        print("[API] Status reset to idle")
        return {
            "code": 0,
            "message": "Status reset"
        }
    except Exception as e:
        print(f"[API] reset_batch_status error: {e}")
        traceback.print_exc()
        return {
            "code": 1,
            "message": str(e)
        }
