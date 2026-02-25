#!/usr/bin/env python3
import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from pymilvus import connections, Collection
import argparse

import yolo_inf
from run_single import load_model, pad_to_square_and_rgb
from dotenv import load_dotenv
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()


# =============================
# CONFIG
# =============================
MILVUS_URI = user_secrets.get_secret("MILVUS_URI")
MILVUS_TOKEN = user_secrets.get_secret("MILVUS_TOKEN")
COLLECTION_NAME = user_secrets.get_secret("COLLECTION_NAME")

VECTOR_DIM = 320
BATCH_SIZE = 128


# =============================
# Feature Extraction
# =============================

def register_hooks(model, features_dict):

    def get_hook(name):
        def hook(module, input, output):
            features_dict[name] = output
        return hook

    model.model[17].register_forward_hook(get_hook("small"))
    model.model[20].register_forward_hook(get_hook("medium"))
    model.model[23].register_forward_hook(get_hook("large"))


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--annotation_csv", required=True)
    parser.add_argument("--img_size", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--iou", type=float, default=0.4)

    opt = parser.parse_args()

    # =============================
    # Connect Milvus
    # =============================
    print("Connecting to Milvus...")
    connections.connect(uri=MILVUS_URI, token=MILVUS_TOKEN)
    collection = Collection(COLLECTION_NAME)

    # =============================
    # Load model
    # =============================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(opt.weights, device)

    features = {}
    register_hooks(model, features)

    stride_map = {0: 8, 1: 16, 2: 32}
    branch_name_map = {0: "small", 1: "medium", 2: "large"}

    # =============================
    # Load image list
    # =============================
    import pandas as pd

    df = pd.read_csv(opt.annotation_csv)

    # Lấy unique image_id
    image_list = df["image_id"].unique().tolist()

    print(f"Total unique images from CSV: {len(image_list)}")

    batch_image_ids = []
    batch_branch_ids = []
    batch_labels = []
    batch_vectors = []

    total_inserted = 0

    # =============================
    # Main Loop
    # =============================
    for img_name in tqdm(image_list):

        img_path = os.path.join(opt.image_dir, img_name)
        if not os.path.exists(img_path):
            img_path = os.path.join(opt.image_dir, img_name + ".png")

        img = cv2.imread(img_path)
        if img is None:
            continue

        padded = pad_to_square_and_rgb(img)

        boxes, scores, labels, branch_ids, boxes_resized = yolo_inf.detect1Image(
            padded,
            opt.img_size,
            model,
            device,
            opt.conf,
            opt.iou
        )

        for i, box in enumerate(boxes_resized):

            branch_id = int(branch_ids[i])
            branch_name = branch_name_map[branch_id]
            stride = stride_map[branch_id]

            x1, y1, x2, y2 = box
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            gx = int(cx / stride)
            gy = int(cy / stride)

            fmap = features[branch_name][0]
            C, H, W = fmap.shape

            gx = np.clip(gx, 0, W - 1)
            gy = np.clip(gy, 0, H - 1)

            vec = fmap[:, gy, gx]
            vec = torch.nn.functional.normalize(vec, dim=0)
            vec = vec.detach().cpu().numpy()

            # Add to batch
            batch_image_ids.append(img_name)
            batch_branch_ids.append(branch_id)
            batch_labels.append(int(labels[i]))
            batch_vectors.append(vec)

        # Insert batch
        if len(batch_vectors) >= BATCH_SIZE:

            data = [
                batch_image_ids,
                batch_labels,
                batch_branch_ids,
                batch_vectors
            ]

            collection.insert(data)
            collection.flush()

            total_inserted += len(batch_vectors)
            print(f"Inserted {total_inserted} vectors")

            batch_image_ids = []
            batch_branch_ids = []
            batch_labels = []
            batch_vectors = []

    # Insert remaining
    if len(batch_vectors) > 0:
        data = [
            batch_image_ids,
            batch_labels,
            batch_branch_ids,
            batch_vectors
        ]
        collection.insert(data)
        collection.flush()
        total_inserted += len(batch_vectors)

    print(f"Finished. Total inserted: {total_inserted}")

    if len(collection.indexes) == 0:
        print("Creating index...")

        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": "IP",
            "params": {"nlist": 1024}
        }

        collection.create_index(
            field_name="embedding",
            index_params=index_params
        )

        print("Index created.")
    else:
        print("Index already exists.")


if __name__ == "__main__":
    main()