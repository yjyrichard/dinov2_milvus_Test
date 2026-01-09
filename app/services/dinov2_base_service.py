"""DINOv2 Base 特征提取服务"""
import torch
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
from typing import Optional
import time

from app.config import DINOV2_BASE_MODEL, EMBEDDING_DIM_BASE
from app.services.image_preprocessor import preprocess_for_dinov2


class DINOv2BaseExtractor:
    _instance: Optional["DINOv2BaseExtractor"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
            cls._instance.processor = None
            cls._instance.model = None
            cls._instance.device = None
        return cls._instance

    def _ensure_initialized(self):
        """确保模型已加载（延迟初始化）"""
        if self._initialized:
            return

        print(f"[DINOV2-BASE] ========== Initializing DINOv2 Base ==========")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[DINOV2-BASE] Device: {self.device}")

        if self.device == "cuda":
            print(f"[DINOV2-BASE] GPU: {torch.cuda.get_device_name(0)}")
            print(f"[DINOV2-BASE] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

        print(f"[DINOV2-BASE] Loading processor from {DINOV2_BASE_MODEL}...")
        self.processor = AutoImageProcessor.from_pretrained(DINOV2_BASE_MODEL)
        print(f"[DINOV2-BASE] Processor loaded")

        print(f"[DINOV2-BASE] Loading model from {DINOV2_BASE_MODEL}...")
        self.model = AutoModel.from_pretrained(DINOV2_BASE_MODEL).to(self.device)
        self.model.eval()
        print(f"[DINOV2-BASE] Model loaded and set to eval mode")

        self._initialized = True
        print(f"[DINOV2-BASE] ========== DINOv2 Base Ready ==========")

    def _load_image(self, image_path: str) -> Image.Image:
        """加载图片，处理 TIF 多页情况"""
        img = Image.open(image_path)
        # TIF 可能是多页的，只取第一页
        if hasattr(img, "n_frames") and img.n_frames > 1:
            img.seek(0)
        return img.convert("RGB")

    def extract_single(self, image_path: str) -> tuple[list[float], float]:
        """
        提取单张图片特征
        返回: (768维向量, 耗时ms)
        """
        self._ensure_initialized()

        start_time = time.time()

        image = self._load_image(image_path)
        # Letterbox Padding 预处理，保持长宽比
        image = preprocess_for_dinov2(image)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            # 使用 CLS token 作为图像特征 (第一个 token)
            features = outputs.last_hidden_state[:, 0, :]
            # L2 归一化
            features = torch.nn.functional.normalize(features, p=2, dim=1)

        elapsed_ms = (time.time() - start_time) * 1000
        return features.squeeze().cpu().tolist(), elapsed_ms

    def extract_from_pil(self, image: Image.Image) -> tuple[list[float], float]:
        """
        从 PIL Image 提取特征（用于上传的图片）
        返回: (768维向量, 耗时ms)
        """
        self._ensure_initialized()

        print(f"[DINOV2-BASE] Extracting features from uploaded image...")
        start_time = time.time()

        image = image.convert("RGB")
        # Letterbox Padding 预处理，保持长宽比
        image = preprocess_for_dinov2(image)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            features = outputs.last_hidden_state[:, 0, :]
            features = torch.nn.functional.normalize(features, p=2, dim=1)

        elapsed_ms = (time.time() - start_time) * 1000
        print(f"[DINOV2-BASE] Feature extraction completed in {elapsed_ms:.1f}ms")
        return features.squeeze().cpu().tolist(), elapsed_ms

    def extract_batch(self, image_paths: list[str]) -> tuple[list[list[float]], float]:
        """
        批量提取特征
        返回: (向量列表, 耗时ms)
        """
        self._ensure_initialized()

        print(f"[DINOV2-BASE] Extracting features from {len(image_paths)} images...")
        start_time = time.time()

        # 加载图片
        load_start = time.time()
        images = []
        for i, p in enumerate(image_paths):
            try:
                img = self._load_image(p)
                # Letterbox Padding 预处理，保持长宽比
                img = preprocess_for_dinov2(img)
                images.append(img)
            except Exception as e:
                print(f"[DINOV2-BASE] Failed to load image {p}: {e}")
                raise
        load_time = (time.time() - load_start) * 1000
        print(f"[DINOV2-BASE] Images loaded in {load_time:.1f}ms")

        # 处理
        process_start = time.time()
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        process_time = (time.time() - process_start) * 1000
        print(f"[DINOV2-BASE] Preprocessing completed in {process_time:.1f}ms")

        # 推理
        infer_start = time.time()
        with torch.no_grad():
            outputs = self.model(**inputs)
            features = outputs.last_hidden_state[:, 0, :]
            features = torch.nn.functional.normalize(features, p=2, dim=1)
        infer_time = (time.time() - infer_start) * 1000
        print(f"[DINOV2-BASE] Inference completed in {infer_time:.1f}ms")

        elapsed_ms = (time.time() - start_time) * 1000

        # 清理 GPU 内存
        if self.device == "cuda":
            torch.cuda.empty_cache()

        print(f"[DINOV2-BASE] Total batch extraction: {elapsed_ms:.1f}ms")
        return features.cpu().tolist(), elapsed_ms

    def get_device_info(self) -> dict:
        """获取设备信息"""
        # 不初始化模型，只返回基本信息
        device = "cuda" if torch.cuda.is_available() else "cpu"

        info = {
            "device": device,
            "model": DINOV2_BASE_MODEL,
            "embedding_dim": EMBEDDING_DIM_BASE,
            "initialized": self._initialized,
        }

        if device == "cuda":
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_total_mb"] = round(
                torch.cuda.get_device_properties(0).total_memory / 1024**2, 2
            )
            if self._initialized:
                info["gpu_memory_allocated_mb"] = round(
                    torch.cuda.memory_allocated() / 1024**2, 2
                )

        return info


# 单例实例（延迟初始化，不在导入时加载模型）
dinov2_base_extractor = DINOv2BaseExtractor()
