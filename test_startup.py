"""启动诊断脚本 - 检查所有依赖是否正常"""
import sys

print("=" * 50)
print("DINOv2 Image Search - 启动诊断")
print("=" * 50)

# 1. 检查 Python 版本
print(f"\n[1] Python 版本: {sys.version}")

# 2. 检查必要的包
print("\n[2] 检查依赖包...")
packages = [
    ("fastapi", "FastAPI"),
    ("uvicorn", "Uvicorn"),
    ("pymilvus", "PyMilvus"),
    ("torch", "PyTorch"),
    ("transformers", "Transformers"),
    ("PIL", "Pillow"),
]

all_ok = True
for pkg_name, display_name in packages:
    try:
        module = __import__(pkg_name)
        version = getattr(module, "__version__", "unknown")
        print(f"   ✓ {display_name}: {version}")
    except ImportError as e:
        print(f"   ✗ {display_name}: 未安装 ({e})")
        all_ok = False

if not all_ok:
    print("\n请运行: pip install -r requirements.txt")
    sys.exit(1)

# 3. 检查 PyTorch GPU
print("\n[3] 检查 GPU...")
import torch
if torch.cuda.is_available():
    print(f"   ✓ CUDA 可用: {torch.cuda.get_device_name(0)}")
else:
    print("   ! CUDA 不可用，将使用 CPU（速度较慢）")

# 4. 检查配置
print("\n[4] 检查配置...")
try:
    from app.config import MILVUS_HOST, MILVUS_PORT, IMAGE_DIR
    print(f"   Milvus: {MILVUS_HOST}:{MILVUS_PORT}")
    print(f"   图片目录: {IMAGE_DIR}")
except ImportError as e:
    print(f"   ✗ 配置加载失败: {e}")
    sys.exit(1)

# 5. 检查图片目录
print("\n[5] 检查图片目录...")
import os
if os.path.exists(IMAGE_DIR):
    tif_files = [f for f in os.listdir(IMAGE_DIR) if f.upper().endswith(".TIF")]
    print(f"   ✓ 目录存在，TIF 文件数: {len(tif_files)}")
else:
    print(f"   ✗ 目录不存在: {IMAGE_DIR}")

# 6. 测试 Milvus 连接
print("\n[6] 测试 Milvus 连接...")
try:
    from pymilvus import connections
    connections.connect(host=MILVUS_HOST, port=MILVUS_PORT, timeout=5)
    print(f"   ✓ Milvus 连接成功")
    connections.disconnect("default")
except Exception as e:
    print(f"   ✗ Milvus 连接失败: {e}")
    print(f"   请检查 Milvus 服务是否在 {MILVUS_HOST}:{MILVUS_PORT} 运行")

# 7. 测试导入
print("\n[7] 测试模块导入...")
try:
    from app.services.milvus_service import milvus_service
    print("   ✓ milvus_service")
except Exception as e:
    print(f"   ✗ milvus_service: {e}")

try:
    from app.services.dinov2_service import dinov2_extractor
    print("   ✓ dinov2_extractor")
except Exception as e:
    print(f"   ✗ dinov2_extractor: {e}")

try:
    from app.services.batch_import_service import batch_import_service
    print("   ✓ batch_import_service")
except Exception as e:
    print(f"   ✗ batch_import_service: {e}")

try:
    from app.api import search, batch, image
    print("   ✓ API routes")
except Exception as e:
    print(f"   ✗ API routes: {e}")

try:
    from main import app
    print("   ✓ FastAPI app")
except Exception as e:
    print(f"   ✗ FastAPI app: {e}")

print("\n" + "=" * 50)
print("诊断完成！如果上述检查全部通过，运行:")
print("  uvicorn main:app --reload --port 8000")
print("=" * 50)
