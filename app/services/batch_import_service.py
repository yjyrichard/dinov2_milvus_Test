"""批量导入服务"""
import os
import re
import json
import time
import asyncio
import threading
from datetime import datetime
from typing import Optional
from PIL import Image
from pathlib import Path

from app.config import (
    IMAGE_DIR,
    THUMBNAIL_DIR,
    BATCH_SIZE,
    MILVUS_INSERT_BATCH
)


def parse_filename(filename: str) -> Optional[dict]:
    """
    解析文件名
    USD1107373-20251230-D00001.TIF -> {'patent_id': 'USD1107373', 'date': '20251230', 'page_num': 'D00001'}
    """
    # 支持带 _1 后缀的文件名
    match = re.match(r"(USD\d+)-(\d+)-(D\d+)(?:_\d+)?\.TIF", filename, re.IGNORECASE)
    if match:
        return {
            "patent_id": match.group(1),
            "date": match.group(2),
            "page_num": match.group(3),
        }
    return None


def generate_thumbnail(src_path: str, size: tuple[int, int] = (300, 300)) -> Optional[str]:
    """生成缩略图"""
    try:
        filename = os.path.basename(src_path)
        name, _ = os.path.splitext(filename)
        dst_path = os.path.join(THUMBNAIL_DIR, f"{name}.jpg")

        # 如果已存在，跳过
        if os.path.exists(dst_path):
            return dst_path

        img = Image.open(src_path)
        if hasattr(img, "n_frames") and img.n_frames > 1:
            img.seek(0)
        img = img.convert("RGB")
        img.thumbnail(size, Image.Resampling.LANCZOS)
        img.save(dst_path, "JPEG", quality=85)

        return dst_path
    except Exception as e:
        print(f"[THUMBNAIL] Failed to generate thumbnail for {src_path}: {e}")
        return None


class BatchImportService:
    """批量导入服务"""

    def __init__(self):
        self.status = "idle"  # idle | running | completed | error
        self.total = 0
        self.processed = 0
        self.failed: list[str] = []
        self.start_time: Optional[float] = None
        self.error_message: Optional[str] = None
        self._thread: Optional[threading.Thread] = None

    def get_status(self) -> dict:
        """获取导入状态"""
        elapsed_sec = 0
        avg_speed = 0
        estimated_remaining_sec = 0

        if self.start_time and self.processed > 0:
            elapsed_sec = time.time() - self.start_time
            avg_speed = self.processed / elapsed_sec
            remaining = self.total - self.processed
            if avg_speed > 0:
                estimated_remaining_sec = remaining / avg_speed

        return {
            "status": self.status,
            "total": self.total,
            "processed": self.processed,
            "failed": len(self.failed),
            "failed_files": self.failed[-10:],  # 最近10个失败文件
            "progress_percent": round(self.processed / self.total * 100, 1) if self.total > 0 else 0,
            "elapsed_sec": round(elapsed_sec, 1),
            "estimated_remaining_sec": round(estimated_remaining_sec, 1),
            "avg_speed": f"{avg_speed:.1f} images/sec" if avg_speed > 0 else "0",
            "error_message": self.error_message,
        }

    def _save_progress(self):
        """保存进度到文件"""
        progress = {
            "status": self.status,
            "total": self.total,
            "processed": self.processed,
            "failed": self.failed,
            "last_update": datetime.now().isoformat(),
        }
        progress_file = os.path.join(IMAGE_DIR, "import_progress.json")
        try:
            with open(progress_file, "w", encoding="utf-8") as f:
                json.dump(progress, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[PROGRESS] Failed to save progress: {e}")

    def _scan_images(self) -> list[str]:
        """扫描图片目录"""
        print(f"[SCAN] Scanning directory: {IMAGE_DIR}")
        images = []
        try:
            for filename in os.listdir(IMAGE_DIR):
                if filename.upper().endswith(".TIF"):
                    images.append(os.path.join(IMAGE_DIR, filename))
            print(f"[SCAN] Found {len(images)} TIF files")
        except Exception as e:
            print(f"[SCAN] Error scanning directory: {e}")
        return sorted(images)

    def start_import(self) -> bool:
        """启动导入任务（在后台线程中运行）"""
        print(f"[START] Current status: {self.status}")

        if self.status == "running":
            print("[START] Already running, skip")
            return False

        self.status = "running"
        self.processed = 0
        self.failed = []
        self.error_message = None
        self.start_time = time.time()

        print("[START] Starting import thread...")

        # 使用线程执行，避免阻塞事件循环
        self._thread = threading.Thread(target=self._do_import, daemon=True)
        self._thread.start()

        print("[START] Import thread started")
        return True

    def _do_import(self):
        """执行导入（在线程中运行）"""
        print("[IMPORT] ========== Import started ==========")

        try:
            # 确保缩略图目录存在
            os.makedirs(THUMBNAIL_DIR, exist_ok=True)
            print(f"[IMPORT] Thumbnail directory: {THUMBNAIL_DIR}")

            # 延迟导入，避免循环依赖
            from app.services.milvus_service import milvus_service
            from app.services.dinov2_service import dinov2_extractor

            # 连接 Milvus
            print("[IMPORT] Connecting to Milvus...")
            milvus_service.connect()
            print("[IMPORT] Creating collection...")
            milvus_service.create_collection()
            print("[IMPORT] Creating index...")
            milvus_service.create_index()
            print("[IMPORT] Loading collection...")
            milvus_service.load_collection()
            print("[IMPORT] Milvus ready")

            # 获取已存在的文件
            print("[IMPORT] Getting existing files...")
            existing_files = milvus_service.get_existing_files()
            print(f"[IMPORT] Existing files in Milvus: {len(existing_files)}")

            # 扫描待处理图片
            all_images = self._scan_images()
            self.total = len(all_images)
            print(f"[IMPORT] Total images to process: {self.total}")

            if self.total == 0:
                print("[IMPORT] No images found, completing...")
                self.status = "completed"
                self._save_progress()
                return

            # 过滤已存在的
            pending_images = [
                img for img in all_images
                if os.path.basename(img) not in existing_files
            ]
            print(f"[IMPORT] Pending images: {len(pending_images)}")

            # 已存在的算作已处理
            self.processed = len(all_images) - len(pending_images)
            print(f"[IMPORT] Already processed: {self.processed}")

            if len(pending_images) == 0:
                print("[IMPORT] All images already imported, completing...")
                self.status = "completed"
                self._save_progress()
                return

            # 分批处理
            total_batches = (len(pending_images) + BATCH_SIZE - 1) // BATCH_SIZE
            print(f"[IMPORT] Total batches: {total_batches}, batch size: {BATCH_SIZE}")

            for batch_idx, i in enumerate(range(0, len(pending_images), BATCH_SIZE)):
                batch_paths = pending_images[i:i + BATCH_SIZE]
                print(f"[BATCH {batch_idx + 1}/{total_batches}] Processing {len(batch_paths)} images...")

                # 提取特征
                embeddings = []
                try:
                    print(f"[BATCH {batch_idx + 1}] Extracting features...")
                    embeddings, extract_time = dinov2_extractor.extract_batch(batch_paths)
                    print(f"[BATCH {batch_idx + 1}] Features extracted in {extract_time:.1f}ms")
                except Exception as e:
                    # 批量失败，改为单张处理
                    print(f"[BATCH {batch_idx + 1}] Batch extraction failed: {e}, falling back to single...")
                    for path in batch_paths:
                        try:
                            emb, _ = dinov2_extractor.extract_single(path)
                            embeddings.append(emb)
                        except Exception as e2:
                            print(f"[BATCH {batch_idx + 1}] Failed to extract {os.path.basename(path)}: {e2}")
                            self.failed.append(os.path.basename(path))
                            embeddings.append(None)

                # 准备数据
                insert_data = []
                for path, embedding in zip(batch_paths, embeddings):
                    if embedding is None:
                        continue

                    filename = os.path.basename(path)
                    parsed = parse_filename(filename)

                    if parsed is None:
                        print(f"[BATCH {batch_idx + 1}] Failed to parse filename: {filename}")
                        self.failed.append(filename)
                        continue

                    # 生成缩略图
                    generate_thumbnail(path)

                    insert_data.append({
                        "file_name": filename,
                        "patent_id": parsed["patent_id"],
                        "page_num": parsed["page_num"],
                        "file_path": path,
                        "embedding": embedding,
                    })

                # 插入 Milvus
                if insert_data:
                    try:
                        print(f"[BATCH {batch_idx + 1}] Inserting {len(insert_data)} records to Milvus...")
                        milvus_service.insert(insert_data)
                        print(f"[BATCH {batch_idx + 1}] Insert successful")
                    except Exception as e:
                        print(f"[BATCH {batch_idx + 1}] Failed to insert batch: {e}")
                        for d in insert_data:
                            self.failed.append(d["file_name"])

                self.processed += len(batch_paths)
                self._save_progress()

                # 打印进度
                progress = self.processed / self.total * 100
                print(f"[PROGRESS] {self.processed}/{self.total} ({progress:.1f}%)")

            # 完成
            self.status = "completed"
            self._save_progress()
            print(f"[IMPORT] ========== Import completed ==========")
            print(f"[IMPORT] Processed: {self.processed}, Failed: {len(self.failed)}")

        except Exception as e:
            self.status = "error"
            self.error_message = str(e)
            self._save_progress()
            print(f"[IMPORT] ========== Import error ==========")
            print(f"[IMPORT] Error: {e}")
            import traceback
            traceback.print_exc()


# 单例实例
batch_import_service = BatchImportService()
