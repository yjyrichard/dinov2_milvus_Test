# MinIO 图片存储迁移方案

## 1. 概述

### 1.1 当前状态
- **testImage 文件夹**: 6387 张图片 (专利图片，TIF 格式为主)
- **睿观文件夹**: 126 张图片 (PNG 格式)
- **存储方式**: 本地文件系统
- **向量数据库**: Milvus (`patent_images_base` collection)

### 1.2 目标状态
- **图片存储**: MinIO 对象存储
- **MinIO 配置**:
  - 地址: `192.168.1.174:9000`
  - Bucket: `trizhi01`
  - Access Key: `admin`
  - Secret Key: `trizhi2026`
- **Milvus Collections**:
  - `patent_images_base`: testImage 图片 (使用 DINOv2-base 模型)
  - `ruiguan_images_base`: 睿观图片 (新 collection，使用 DINOv2-base 模型)

## 2. 架构变更

### 2.1 图片存储路径变化

| 变更前 | 变更后 |
|--------|--------|
| 本地路径: `testImage/xxx.TIF` | MinIO URL: `http://192.168.1.174:9000/trizhi01/testImage/xxx.TIF` |
| 本地路径: `睿观/xxx.png` | MinIO URL: `http://192.168.1.174:9000/trizhi01/ruiguan/xxx.png` |

### 2.2 MinIO Bucket 结构
```
trizhi01/
├── testImage/           # 专利图片
│   ├── USD1107373-20251230-D00000.TIF
│   ├── USD1107373-20251230-D00001.TIF
│   └── ...
├── ruiguan/            # 睿观图片
│   ├── 睿观(1).png
│   ├── 睿观(2).png
│   └── ...
└── thumbnails/         # 缩略图 (可选)
    ├── testImage/
    └── ruiguan/
```

## 3. 代码变更清单

### 3.1 新增文件

| 文件 | 说明 |
|------|------|
| `app/services/minio_service.py` | MinIO 服务封装 (上传、下载、URL生成) |
| `app/services/milvus_ruiguan_service.py` | 睿观专用 Milvus 服务 |
| `scripts/upload_testimage_to_minio.py` | testImage 上传脚本 |
| `scripts/upload_ruiguan_to_minio.py` | 睿观上传脚本 (独立) |
| `scripts/import_ruiguan_to_milvus.py` | 睿观导入 Milvus 脚本 |

### 3.2 修改文件

| 文件 | 变更内容 |
|------|----------|
| `app/config.py` | 新增 MinIO 配置、睿观 collection 配置 |
| `app/api/image.py` | 从 MinIO 获取图片，支持代理模式 |
| `app/services/batch_import_base_service.py` | 导入时上传到 MinIO，存储 MinIO URL |
| `app/services/milvus_base_service.py` | file_path 字段存储 MinIO URL |

### 3.3 配置变更 (config.py)

```python
# 新增 MinIO 配置
MINIO_ENDPOINT = "192.168.1.174:9000"
MINIO_ACCESS_KEY = "admin"
MINIO_SECRET_KEY = "trizhi2026"
MINIO_BUCKET = "trizhi01"
MINIO_SECURE = False  # HTTP

# 新增睿观 collection 配置
COLLECTION_NAME_RUIGUAN = "ruiguan_images_base"
```

## 4. 迁移步骤

### 4.1 准备工作
```bash
# 1. 安装 MinIO Python SDK
pip install minio

# 2. 验证 MinIO 连接
python -c "from minio import Minio; c = Minio('192.168.1.174:9000', 'admin', 'trizhi2026', secure=False); print(c.bucket_exists('trizhi01'))"
```

### 4.2 上传 testImage 到 MinIO
```bash
# 运行上传脚本 (支持断点续传)
python scripts/upload_testimage_to_minio.py

# 输出示例:
# [UPLOAD] Starting upload to MinIO...
# [UPLOAD] Progress: 1000/6387 (15.7%)
# [UPLOAD] Progress: 2000/6387 (31.3%)
# ...
# [UPLOAD] Completed: 6387 files uploaded
```

### 4.3 重新导入到 Milvus (可选)
如果需要更新 Milvus 中的 file_path 为 MinIO URL：
```bash
# 方式1: 删除旧 collection，重新导入
python scripts/reimport_to_milvus.py --drop-existing

# 方式2: 保留现有数据，只更新 file_path (需要自定义脚本)
```

### 4.4 上传睿观图片到 MinIO
```bash
# 运行独立上传脚本
python scripts/upload_ruiguan_to_minio.py

# 输出示例:
# [RUIGUAN] Uploading 126 images to MinIO...
# [RUIGUAN] Completed: 126 files uploaded
```

### 4.5 导入睿观到新 Collection
```bash
# 创建 ruiguan_images_base collection 并导入
python scripts/import_ruiguan_to_milvus.py

# 输出示例:
# [RUIGUAN] Creating collection 'ruiguan_images_base'...
# [RUIGUAN] Processing 126 images...
# [RUIGUAN] Completed: 126 vectors inserted
```

## 5. 启动方式

### 5.1 完整迁移流程
```bash
# 步骤1: 安装依赖
pip install minio

# 步骤2: 上传 testImage 到 MinIO
python scripts/upload_testimage_to_minio.py

# 步骤3: 上传睿观到 MinIO
python scripts/upload_ruiguan_to_minio.py

# 步骤4: 导入睿观到 Milvus
python scripts/import_ruiguan_to_milvus.py

# 步骤5: 启动服务
python main.py
```

### 5.2 API 变更

#### 图片访问 (保持兼容)
```
# 原有接口保持不变，后端自动从 MinIO 获取
GET /api/image/{file_name}        # 缩略图
GET /api/image/full/{file_name}   # 原图
```

#### 搜索结果变化
```json
// 变更前
{
  "file_path": "testImage/USD1107373-20251230-D00001.TIF"
}

// 变更后
{
  "file_path": "http://192.168.1.174:9000/trizhi01/testImage/USD1107373-20251230-D00001.TIF",
  "minio_url": "http://192.168.1.174:9000/trizhi01/testImage/USD1107373-20251230-D00001.TIF"
}
```

## 6. 优势

1. **可扩展性**: MinIO 支持分布式部署，存储容量可横向扩展
2. **高可用**: 支持多副本，数据更安全
3. **统一访问**: 所有图片通过 HTTP URL 访问，前端可直接使用
4. **备份方便**: MinIO 支持版本控制和跨区域复制
5. **独立部署**: 图片存储与应用服务器分离，便于维护

## 7. 注意事项

1. **网络带宽**: 首次上传 6387 张图片需要一定时间，建议在业务低峰期执行
2. **MinIO 权限**: 确保 Bucket 策略允许读取访问
3. **断点续传**: 上传脚本支持断点续传，中断后可继续
4. **缩略图**: 可选择是否将缩略图也上传到 MinIO

## 8. 回滚方案

如果迁移过程中出现问题：
1. 本地图片保持不动，不会删除
2. Milvus 数据可以重新导入
3. 配置切换回本地模式即可恢复

---

**确认后我将开始实现上述代码变更。**
