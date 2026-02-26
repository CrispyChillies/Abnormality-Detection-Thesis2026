#!/usr/bin/env python3
import os
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pymilvus import connections, Collection
import argparse

import yolo_inf
from run_single import load_model, pad_to_square_and_rgb
from dotenv import load_dotenv


# =============================
# IoU
# =============================

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0


# =============================
# Hook
# =============================

def register_hooks(model, features_dict):

    def get_hook(name):
        def hook(module, input, output):
            features_dict[name] = output
        return hook

    model.model[17].register_forward_hook(get_hook("small"))
    model.model[20].register_forward_hook(get_hook("medium"))
    model.model[23].register_forward_hook(get_hook("large"))


# =============================
# Main
# =============================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--annotation_csv", required=True)
    parser.add_argument("--img_size", type=int, default=384)
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--iou", type=float, default=0.4)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--metric", type=str, default="IP")  # IP or L2
    opt = parser.parse_args()

    load_dotenv()
    MILVUS_URI = os.getenv("MILVUS_URI")
    MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME")

    print("Connecting to Milvus...")
    connections.connect(uri=MILVUS_URI, token=MILVUS_TOKEN)
    collection = Collection(COLLECTION_NAME)
    collection.load()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(opt.weights, device)

    features = {}
    register_hooks(model, features)

    stride_map = {0: 8, 1: 16, 2: 32}
    branch_name_map = {0: "small", 1: "medium", 2: "large"}

    print("Loading test annotation...")
    df = pd.read_csv(opt.annotation_csv)

    gt_dict = {}
    for _, row in df.iterrows():
        img = row["image_id"]
        box = [row["x_min"], row["y_min"], row["x_max"], row["y_max"]]
        label = int(row["class_id"]) if "class_id" in row else row["class_name"]

        if img not in gt_dict:
            gt_dict[img] = []
        gt_dict[img].append({"bbox": box, "label": label})

    image_list = df["image_id"].unique().tolist()

    search_params = {
        "metric_type": opt.metric,
        "params": {"nprobe": 10}
    }

    rp1_list = []
    rp5_list = []
    cc1_list = []
    cc5_list = []

    print("Starting evaluation...")

    for img_name in tqdm(image_list):

        img_path = os.path.join(opt.image_dir, img_name)
        if not os.path.exists(img_path):
            img_path = os.path.join(opt.image_dir, img_name + ".png")

        img = cv2.imread(img_path)
        if img is None:
            continue

        padded = pad_to_square_and_rgb(img)

        boxes, scores, labels, branch_ids, boxes_resized = yolo_inf.detect1Image(
            padded, opt.img_size, model, device, opt.conf, opt.iou
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

            results = collection.search(
                data=[vec],
                anns_field="embedding",
                param=search_params,
                limit=opt.topk,
                output_fields=["label"]
            )

            retrieved = results[0]

            query_label = int(labels[i])

            correct1 = 1 if retrieved[0].entity.get("label") == query_label else 0
            correct5 = sum(
                1 for r in retrieved if r.entity.get("label") == query_label
            )

            rp1_list.append(correct1)
            rp5_list.append(correct5 / opt.topk)

            # CC only if matched GT
            if img_name in gt_dict:
                matched = False
                for gt in gt_dict[img_name]:
                    if compute_iou(box, gt["bbox"]) >= 0.4:
                        matched = True
                        break

                if matched:
                    cc1_list.append(correct1)
                    cc5_list.append(correct5 / opt.topk)

    print("===== FINAL RESULTS =====")
    print("RP@1:", np.mean(rp1_list))
    print("RP@5:", np.mean(rp5_list))

    if len(cc1_list) > 0:
        print("CC@1:", np.mean(cc1_list))
        print("CC@5:", np.mean(cc5_list))
    else:
        print("No GT matched lesions for CC metric.")


if __name__ == "__main__":
    main()