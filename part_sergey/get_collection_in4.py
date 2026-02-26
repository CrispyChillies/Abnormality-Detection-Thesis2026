from pymilvus import connections, Collection
from dotenv import load_dotenv
import os
from collections import Counter

load_dotenv()
MILVUS_URI = os.getenv("MILVUS_URI")
print(MILVUS_URI)
MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")

connections.connect(uri=MILVUS_URI, token=MILVUS_TOKEN)
collection = Collection(COLLECTION_NAME)

# Load collection (quan trọng trước khi query)
collection.load()

# 1️⃣ Total lesions
print("Total lesions in library:", collection.num_entities)

# 2️⃣ Unique image_ids
image_results = collection.query(
    expr="image_id != ''",
    output_fields=["image_id"]
)

unique_image_ids = set([r["image_id"] for r in image_results])
print("Unique image_ids:", len(unique_image_ids))

# 3️⃣ Unique labels
label_results = collection.query(
    expr="label >= 0",
    output_fields=["label"]
)

unique_labels = set([r["label"] for r in label_results])
print("Unique labels:", len(unique_labels))
print("Label list:", sorted(unique_labels))

branch_results = collection.query(
    expr="branch_id >= 0",
    output_fields=["branch_id"]
)

counter = Counter([r["branch_id"] for r in branch_results])
print(counter)

label_results = collection.query(
    expr="label >= 0",
    output_fields=["label"]
)

from collections import Counter
label_counter = Counter([r["label"] for r in label_results])

for k,v in sorted(label_counter.items()):
    print(k, v)