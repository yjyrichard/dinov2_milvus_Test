"""图像预处理模块 - Letterbox Padding

解决问题：
- 直接 Resize 会破坏长宽比，导致细长物体被压扁
- 高分辨率图片缩小后细节丢失

方案：
- Letterbox Padding: 保持长宽比缩放，用白色填充到目标尺寸
"""
from PIL import Image


def letterbox_resize(image: Image.Image, target_size: int = 518) -> Image.Image:
    """
    Letterbox Padding 预处理

    1. 保持长宽比，将长边缩放到 target_size
    2. 创建白色背景 (target_size x target_size)
    3. 将缩放后的图片居中贴上

    Args:
        image: PIL Image (RGB)
        target_size: 目标尺寸，默认 518 (DINOv2 推荐)

    Returns:
        处理后的 PIL Image (target_size x target_size)
    """
    # 确保是 RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    orig_width, orig_height = image.size

    # 计算缩放比例，保持长宽比
    scale = target_size / max(orig_width, orig_height)
    new_width = int(orig_width * scale)
    new_height = int(orig_height * scale)

    # 缩放图片（使用高质量重采样）
    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # 创建白色背景
    background = Image.new("RGB", (target_size, target_size), (255, 255, 255))

    # 计算居中位置
    paste_x = (target_size - new_width) // 2
    paste_y = (target_size - new_height) // 2

    # 贴上缩放后的图片
    background.paste(resized, (paste_x, paste_y))

    return background


def preprocess_for_dinov2(image: Image.Image, target_size: int = 518) -> Image.Image:
    """
    DINOv2 专用预处理入口

    当前实现: Letterbox Padding
    后续可扩展: 二值化增强、切片处理等

    Args:
        image: PIL Image
        target_size: 目标尺寸

    Returns:
        预处理后的图片
    """
    return letterbox_resize(image, target_size)
