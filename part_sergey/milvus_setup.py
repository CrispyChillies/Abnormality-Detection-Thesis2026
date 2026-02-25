import os

from dotenv import load_dotenv
from pymilvus import (
    connections,
    FieldSchema, CollectionSchema, DataType,
    Collection
)

# ====== CONFIG ======

load_dotenv()

MILVUS_URI = os.getenv("MILVUS_URI")
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
VECTOR_DIM = 320  # hoặc dim thực tế của bạn
# ====================

connections.connect(
    uri=MILVUS_URI,
    token=MILVUS_TOKEN
)

fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),

    FieldSchema(name="image_id", dtype=DataType.VARCHAR, max_length=100),

    FieldSchema(name="label", dtype=DataType.INT64),

    FieldSchema(name="branch_id", dtype=DataType.INT64),

    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=320),
]

schema = CollectionSchema(fields)

collection = Collection(
    name="vindr_lesion_features",
    schema=schema
)

print("Collection created.")