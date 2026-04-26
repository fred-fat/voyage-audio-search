"""
创建 MongoDB Atlas 向量搜索索引脚本。
幂等：索引已存在时跳过创建。

用法：
    venv/bin/python backend/scripts/create_index.py
"""

import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from pymongo import MongoClient
from pymongo.errors import OperationFailure

from backend.config_service import get_effective_config

DB_NAME = "voyage_audio_search"
COLLECTION_NAME = "audio_records"
INDEX_NAME = "embedding_index"


def create_vector_index() -> None:
    config = get_effective_config()
    mongodb_uri = config.get("mongodb_uri")

    if not mongodb_uri:
        print("错误：未配置 MONGODB_URI，请在 config.local.json 或环境变量中设置。")
        sys.exit(1)

    client = MongoClient(mongodb_uri)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]

    # 检查索引是否已存在
    existing_indexes = list(collection.list_search_indexes())
    for idx in existing_indexes:
        if idx.get("name") == INDEX_NAME:
            print(f"索引 '{INDEX_NAME}' 已存在，跳过创建。")
            client.close()
            return

    # Vector search index definition — fixed at 1024 dims to match voyage-4-large
    index_definition = {
        "fields": [
            {
                "type": "vector",
                "path": "embedding",
                "numDimensions": 1024,
                "similarity": "cosine",
            }
        ]
    }

    search_index_model = {
        "definition": index_definition,
        "name": INDEX_NAME,
        "type": "vectorSearch",
    }

    try:
        collection.create_search_index(model=search_index_model)
        print(f"向量搜索索引 '{INDEX_NAME}' 创建成功。")
        print("注意：Atlas 索引构建需要几分钟，请稍后再使用搜索功能。")
    except OperationFailure as e:
        print(f"创建索引失败：{e}")
        sys.exit(1)
    finally:
        client.close()


if __name__ == "__main__":
    create_vector_index()
