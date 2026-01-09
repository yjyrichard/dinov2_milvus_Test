"""图片服务 API"""
import os
import io
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image

from app.config import IMAGE_DIRS, THUMBNAIL_DIR

router = APIRouter(prefix="/api/image", tags=["image"])


def find_image(file_name: str) -> str | None:
    """从多个目录查找图片"""
    for image_dir in IMAGE_DIRS:
        path = os.path.join(image_dir, file_name)
        if os.path.exists(path):
            return path
    return None


@router.get("/{file_name}")
async def get_thumbnail(file_name: str):
    """获取缩略图"""
    # 去掉扩展名，换成 .jpg
    name, _ = os.path.splitext(file_name)
    thumbnail_path = os.path.join(THUMBNAIL_DIR, f"{name}.jpg")

    if os.path.exists(thumbnail_path):
        return FileResponse(thumbnail_path, media_type="image/jpeg")

    # 如果缩略图不存在，尝试从多个目录查找原图
    original_path = find_image(file_name)
    if not original_path:
        raise HTTPException(status_code=404, detail="Image not found")

    # 实时转换
    try:
        img = Image.open(original_path)
        if hasattr(img, "n_frames") and img.n_frames > 1:
            img.seek(0)
        img = img.convert("RGB")
        img.thumbnail((300, 300), Image.Resampling.LANCZOS)

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        buffer.seek(0)

        return StreamingResponse(buffer, media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process image: {e}")


@router.get("/full/{file_name}")
async def get_full_image(file_name: str):
    """获取原图（转换为 JPEG）"""
    original_path = find_image(file_name)

    if not original_path:
        raise HTTPException(status_code=404, detail="Image not found")

    try:
        img = Image.open(original_path)
        if hasattr(img, "n_frames") and img.n_frames > 1:
            img.seek(0)
        img = img.convert("RGB")

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=95)
        buffer.seek(0)

        return StreamingResponse(buffer, media_type="image/jpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process image: {e}")
