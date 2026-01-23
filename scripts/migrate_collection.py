"""
Milvus Collection 迁移脚本
1. 导出现有数据到 CSV（不含 embedding）
2. 创建新 Collection（VARCHAR 字段扩大到 4096）
3. 迁移数据到新 Collection
4. 完善的错误处理和日志记录
"""
import os
import sys
import csv
import json
import time
import logging
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pymilvus import (
    connections, Collection, FieldSchema, CollectionSchema, DataType, utility
)
from app.config import MILVUS_HOST, MILVUS_PORT, EMBEDDING_DIM_BASE

# 配置
OLD_COLLECTION = "design_patents_full"
NEW_COLLECTION = "design_patents_full_v2"
BATCH_SIZE = 500  # 查询批次
INSERT_BATCH_SIZE = 8  # 插入批次（避免 gRPC 64MB 限制）
MAX_VARCHAR_LEN = 4096  # VARCHAR 最大长度

# 日志配置
LOG_DIR = Path(__file__).parent
LOG_FILE = LOG_DIR / f"migrate_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
CSV_FILE = LOG_DIR / f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 新 Schema 定义（VARCHAR 字段扩大到 4096）
NEW_SCHEMA_FIELDS = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
    FieldSchema(name="patent_id", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="image_index", dtype=DataType.INT16),
    FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=1024),
    FieldSchema(name="file_path", dtype=DataType.VARCHAR, max_length=2048),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM_BASE),
    FieldSchema(name="title", dtype=DataType.VARCHAR, max_length=MAX_VARCHAR_LEN),
    FieldSchema(name="loc_class", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="loc_edition", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="pub_date", dtype=DataType.INT64),
    FieldSchema(name="filing_date", dtype=DataType.INT64),
    FieldSchema(name="grant_term", dtype=DataType.INT16),
    FieldSchema(name="applicant_name", dtype=DataType.VARCHAR, max_length=MAX_VARCHAR_LEN),
    FieldSchema(name="applicant_country", dtype=DataType.VARCHAR, max_length=256),
    FieldSchema(name="inventor_names", dtype=DataType.VARCHAR, max_length=MAX_VARCHAR_LEN),
    FieldSchema(name="assignee_name", dtype=DataType.VARCHAR, max_length=MAX_VARCHAR_LEN),
    FieldSchema(name="claim_text", dtype=DataType.VARCHAR, max_length=MAX_VARCHAR_LEN),
    FieldSchema(name="image_count", dtype=DataType.INT16),
    FieldSchema(name="created_at", dtype=DataType.INT64),
]

INDEX_PARAMS = {
    "index_type": "IVF_FLAT",
    "metric_type": "COSINE",
    "params": {"nlist": 128}
}

# 需要导出的字段（不含 embedding，太大了）
EXPORT_FIELDS = [
    "patent_id", "image_index", "file_name", "file_path",
    "title", "loc_class", "loc_edition", "pub_date", "filing_date",
    "grant_term", "applicant_name", "applicant_country", "inventor_names",
    "assignee_name", "claim_text", "image_count", "created_at"
]

# 迁移时需要的所有字段（含 embedding）
MIGRATE_FIELDS = EXPORT_FIELDS + ["embedding"]


def connect_milvus():
    """连接 Milvus"""
    logger.info(f"连接 Milvus: {MILVUS_HOST}:{MILVUS_PORT}")
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
    logger.info("Milvus 连接成功")


def get_total_count(collection: Collection) -> int:
    """获取 collection 总记录数"""
    collection.flush()
    return collection.num_entities


def export_to_csv(collection: Collection) -> int:
    """导出数据到 CSV（不含 embedding），使用主键游标分页"""
    logger.info(f"开始导出数据到 CSV: {CSV_FILE}")

    total = get_total_count(collection)
    logger.info(f"总记录数: {total}")

    exported = 0
    errors = 0
    last_id = -1  # 游标：上一批最后一条记录的 id

    with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['id'] + EXPORT_FIELDS)
        writer.writeheader()

        while True:
            try:
                results = collection.query(
                    expr=f"id > {last_id}",
                    output_fields=['id'] + EXPORT_FIELDS,
                    limit=BATCH_SIZE
                )

                if not results:
                    break

                for row in results:
                    try:
                        writer.writerow(row)
                        exported += 1
                    except Exception as e:
                        errors += 1
                        logger.error(f"写入 CSV 失败 (id={row.get('id')}): {e}")

                last_id = max(r['id'] for r in results)
                logger.info(f"导出进度: {exported}/{total} ({exported} 成功, {errors} 失败)")

            except Exception as e:
                logger.error(f"查询失败 (last_id={last_id}): {e}")
                break

    logger.info(f"CSV 导出完成: {exported} 条成功, {errors} 条失败")
    return exported


def create_new_collection() -> Collection:
    """创建新 Collection"""
    logger.info(f"创建新 Collection: {NEW_COLLECTION}")

    if utility.has_collection(NEW_COLLECTION):
        logger.warning(f"Collection {NEW_COLLECTION} 已存在，将删除重建")
        utility.drop_collection(NEW_COLLECTION)

    schema = CollectionSchema(NEW_SCHEMA_FIELDS, description="Design patents with extended VARCHAR fields")
    collection = Collection(name=NEW_COLLECTION, schema=schema)

    logger.info("创建索引...")
    collection.create_index(field_name="embedding", index_params=INDEX_PARAMS)
    logger.info("索引创建完成")

    return collection


def truncate_str(s, max_len: int) -> str:
    """安全截断字符串"""
    if s is None:
        return ""
    s = str(s)
    return s[:max_len] if len(s) > max_len else s


def migrate_data(old_collection: Collection, new_collection: Collection) -> tuple:
    """迁移数据，使用主键游标分页"""
    total = get_total_count(old_collection)
    logger.info(f"开始迁移数据: {total} 条记录")

    migrated = 0
    skipped = 0
    errors = 0
    error_ids = []

    batch_data = []
    last_id = -1  # 游标

    while True:
        try:
            # 查询一批数据（含 embedding）
            results = old_collection.query(
                expr=f"id > {last_id}",
                output_fields=['id'] + MIGRATE_FIELDS,
                limit=BATCH_SIZE
            )

            if not results:
                break

            for row in results:
                try:
                    # 准备数据，确保字符串不超长
                    record = {
                        "patent_id": truncate_str(row.get("patent_id"), 256),
                        "image_index": row.get("image_index", 0),
                        "file_name": truncate_str(row.get("file_name"), 1024),
                        "file_path": truncate_str(row.get("file_path"), 2048),
                        "embedding": row.get("embedding"),
                        "title": truncate_str(row.get("title"), MAX_VARCHAR_LEN),
                        "loc_class": truncate_str(row.get("loc_class"), 256),
                        "loc_edition": truncate_str(row.get("loc_edition"), 256),
                        "pub_date": row.get("pub_date", 0),
                        "filing_date": row.get("filing_date", 0),
                        "grant_term": row.get("grant_term", 0),
                        "applicant_name": truncate_str(row.get("applicant_name"), MAX_VARCHAR_LEN),
                        "applicant_country": truncate_str(row.get("applicant_country"), 256),
                        "inventor_names": truncate_str(row.get("inventor_names"), MAX_VARCHAR_LEN),
                        "assignee_name": truncate_str(row.get("assignee_name"), MAX_VARCHAR_LEN),
                        "claim_text": truncate_str(row.get("claim_text"), MAX_VARCHAR_LEN),
                        "image_count": row.get("image_count", 0),
                        "created_at": row.get("created_at", int(time.time())),
                    }

                    # 检查 embedding 是否有效
                    if not record["embedding"] or len(record["embedding"]) != EMBEDDING_DIM_BASE:
                        logger.warning(f"跳过无效 embedding: patent_id={record['patent_id']}, file_name={record['file_name']}")
                        skipped += 1
                        continue

                    batch_data.append(record)

                    # 批量插入
                    if len(batch_data) >= INSERT_BATCH_SIZE:
                        success = insert_batch(new_collection, batch_data)
                        if success:
                            migrated += len(batch_data)
                        else:
                            errors += len(batch_data)
                            for r in batch_data:
                                error_ids.append(f"{r['patent_id']}_{r['file_name']}")
                        batch_data = []

                except Exception as e:
                    errors += 1
                    error_ids.append(f"{row.get('patent_id')}_{row.get('file_name')}")
                    logger.error(f"处理记录失败: {e}")

            last_id = max(r['id'] for r in results)
            logger.info(f"迁移进度: {migrated + skipped + errors}/{total} | 成功: {migrated} | 跳过: {skipped} | 失败: {errors}")

        except Exception as e:
            logger.error(f"查询批次失败 (last_id={last_id}): {e}")
            break

    # 插入剩余数据
    if batch_data:
        success = insert_batch(new_collection, batch_data)
        if success:
            migrated += len(batch_data)
        else:
            errors += len(batch_data)

    # 记录失败的 ID
    if error_ids:
        error_file = LOG_DIR / f"error_ids_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(error_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(error_ids))
        logger.info(f"失败记录 ID 已保存到: {error_file}")

    return migrated, skipped, errors


def insert_batch(collection: Collection, batch_data: list) -> bool:
    """批量插入数据"""
    try:
        entities = [
            [d["patent_id"] for d in batch_data],
            [d["image_index"] for d in batch_data],
            [d["file_name"] for d in batch_data],
            [d["file_path"] for d in batch_data],
            [d["embedding"] for d in batch_data],
            [d["title"] for d in batch_data],
            [d["loc_class"] for d in batch_data],
            [d["loc_edition"] for d in batch_data],
            [d["pub_date"] for d in batch_data],
            [d["filing_date"] for d in batch_data],
            [d["grant_term"] for d in batch_data],
            [d["applicant_name"] for d in batch_data],
            [d["applicant_country"] for d in batch_data],
            [d["inventor_names"] for d in batch_data],
            [d["assignee_name"] for d in batch_data],
            [d["claim_text"] for d in batch_data],
            [d["image_count"] for d in batch_data],
            [d["created_at"] for d in batch_data],
        ]
        collection.insert(entities)
        return True
    except Exception as e:
        logger.error(f"批量插入失败 ({len(batch_data)} 条): {e}")
        # 尝试逐条插入
        success_count = 0
        for d in batch_data:
            try:
                single = [[v] for v in [
                    d["patent_id"], d["image_index"], d["file_name"], d["file_path"],
                    d["embedding"], d["title"], d["loc_class"], d["loc_edition"],
                    d["pub_date"], d["filing_date"], d["grant_term"], d["applicant_name"],
                    d["applicant_country"], d["inventor_names"], d["assignee_name"],
                    d["claim_text"], d["image_count"], d["created_at"]
                ]]
                collection.insert(single)
                success_count += 1
            except Exception as e2:
                logger.error(f"单条插入失败 ({d['patent_id']}_{d['file_name']}): {e2}")
        return success_count == len(batch_data)


def rename_collections():
    """重命名 collection（可选）"""
    # Milvus 不支持直接重命名，需要手动操作
    # 这里只是提示用户
    logger.info("=" * 60)
    logger.info("迁移完成后，如需替换原 collection，请手动执行：")
    logger.info(f"1. 删除旧 collection: utility.drop_collection('{OLD_COLLECTION}')")
    logger.info(f"2. 使用新 collection: {NEW_COLLECTION}")
    logger.info("或者修改代码中的 COLLECTION_NAME 为新名称")
    logger.info("=" * 60)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Milvus Collection 迁移工具')
    parser.add_argument('--export-only', action='store_true', help='仅导出 CSV，不迁移')
    parser.add_argument('--skip-export', action='store_true', help='跳过 CSV 导出')
    args = parser.parse_args()

    start_time = time.time()

    logger.info("=" * 60)
    logger.info("Milvus Collection 迁移工具")
    logger.info(f"源 Collection: {OLD_COLLECTION}")
    logger.info(f"目标 Collection: {NEW_COLLECTION}")
    logger.info(f"日志文件: {LOG_FILE}")
    logger.info("=" * 60)

    try:
        # 连接
        connect_milvus()

        # 检查源 collection
        if not utility.has_collection(OLD_COLLECTION):
            logger.error(f"源 Collection '{OLD_COLLECTION}' 不存在!")
            return

        old_collection = Collection(OLD_COLLECTION)
        old_collection.load()

        total = get_total_count(old_collection)
        logger.info(f"源 Collection 记录数: {total}")

        # 导出 CSV
        if not args.skip_export:
            export_to_csv(old_collection)
            logger.info(f"CSV 文件: {CSV_FILE}")

        if args.export_only:
            logger.info("仅导出模式，跳过迁移")
            return

        # 创建新 collection
        new_collection = create_new_collection()
        new_collection.load()

        # 迁移数据
        migrated, skipped, errors = migrate_data(old_collection, new_collection)

        # 完成
        new_collection.flush()
        new_total = get_total_count(new_collection)

        elapsed = time.time() - start_time

        logger.info("=" * 60)
        logger.info("迁移完成!")
        logger.info(f"源 Collection: {total} 条")
        logger.info(f"成功迁移: {migrated} 条")
        logger.info(f"跳过: {skipped} 条")
        logger.info(f"失败: {errors} 条")
        logger.info(f"新 Collection: {new_total} 条")
        logger.info(f"耗时: {elapsed/60:.1f} 分钟")
        logger.info(f"日志文件: {LOG_FILE}")
        logger.info("=" * 60)

        rename_collections()

    except Exception as e:
        logger.exception(f"迁移失败: {e}")
        raise


if __name__ == "__main__":
    main()
