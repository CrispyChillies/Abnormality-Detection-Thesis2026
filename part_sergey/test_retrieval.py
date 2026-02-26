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
from kaggle_secrets import UserSecretsClient


# =============================
# MAPPING DICTIONARIES
# =============================
NAME_TO_CLASS_ID = {
    "Aortic enlargement": 0, "Atelectasis": 1, "Calcification": 2,
    "Cardiomegaly": 3, "Consolidation": 4, "ILD": 5, "Infiltration": 6,
    "Lung Opacity": 7, "Nodule/Mass": 8, "Other lesion": 9,
    "Pleural effusion": 10, "Pleural thickening": 11, "Pneumothorax": 12,
    "Pulmonary fibrosis": 13
}

CLASS_TO_GROUP = {
    0: "AoE", 1: "PaL", 2: "PaL", 3: "Cmg", 4: "PaL", 5: "PaL",
    6: "PaL", 7: "PaL", 8: "PaL", 9: "OtL", 10: "PlL", 11: "PlL",
    12: "Pnm", 13: "PaL"
}

# Ánh xạ ID sang tên viết tắt để in giống biểu đồ trong paper
CLASS_ID_TO_ABBR = {
    1: "Atl", 2: "Clc", 4: "Cns", 5: "ILD", 6: "Inf", 
    7: "LOp", 8: "Nod", 13: "PuF", 10: "PlE", 11: "PlT"
}

TARGET_CC_GROUPS = ["PaL", "PlL"]
ALL_GROUPS = ["AoE", "Cmg", "PaL", "OtL", "PlL", "Pnm"]

# =============================
# IoU & Hooks
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
    parser.add_argument("--metric", type=str, default="IP") 
    opt = parser.parse_args()

    user_secrets = UserSecretsClient()
    MILVUS_URI = user_secrets.get_secret("MILVUS_URI")
    MILVUS_TOKEN = user_secrets.get_secret("MILVUS_TOKEN")
    COLLECTION_NAME = user_secrets.get_secret("COLLECTION_NAME")

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
        
        if "class_id" in row and pd.notna(row["class_id"]):
            label = int(row["class_id"])
            if label == 14: # Bỏ qua No finding
                continue
        else:
            class_name = str(row["class_name"]).strip()
            if class_name == "No finding":
                continue
            label = NAME_TO_CLASS_ID.get(class_name, -1) 

        if label == -1:
            continue

        if img not in gt_dict:
            gt_dict[img] = []
        gt_dict[img].append({"bbox": box, "label": label})

    image_list = df["image_id"].unique().tolist()
    search_params = {"metric_type": opt.metric, "params": {"nprobe": 10}}

    # TẠO DICTIONARY ĐỂ LƯU KẾT QUẢ THEO TỪNG NHÓM VÀ LỚP
    rp5_per_group = {g: [] for g in ALL_GROUPS}
    cc5_per_class = {c: [] for c in CLASS_ID_TO_ABBR.keys()}

    # Biến lưu tổng thể
    rp1_list, rp5_list = [], []
    cc1_list, cc5_list = [], []

    print("Starting evaluation...")

    for img_name in tqdm(image_list):
        img_path = os.path.join(opt.image_dir, img_name)
        if not os.path.exists(img_path):
            img_path = os.path.join(opt.image_dir, img_name + ".png")

        img = cv2.imread(img_path)
        if img is None: continue

        padded = pad_to_square_and_rgb(img)
        boxes, scores, labels, branch_ids, boxes_resized = yolo_inf.detect1Image(
            padded, opt.img_size, model, device, opt.conf, opt.iou
        )

        for i, box in enumerate(boxes_resized):
            
            # 1. TÌM GROUND TRUTH
            matched_gt_label = None
            max_box_iou = 0
            
            if img_name in gt_dict:
                for gt in gt_dict[img_name]:
                    iou_val = compute_iou(box, gt["bbox"])
                    if iou_val >= 0.4 and iou_val > max_box_iou:
                        max_box_iou = iou_val
                        matched_gt_label = gt["label"]

            if matched_gt_label is None:
                continue

            gt_class = int(matched_gt_label)
            gt_group = CLASS_TO_GROUP.get(gt_class)

            # 2. TRÍCH XUẤT FEATURE VÀ MILVUS
            branch_id = int(branch_ids[i])
            branch_name = branch_name_map[branch_id]
            stride = stride_map[branch_id]
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            gx, gy = int(cx / stride), int(cy / stride)

            fmap = features[branch_name][0]
            C, H, W = fmap.shape
            gx, gy = np.clip(gx, 0, W - 1), np.clip(gy, 0, H - 1)

            vec = fmap[:, gy, gx]
            vec = torch.nn.functional.normalize(vec, dim=0).detach().cpu().numpy()

            results = collection.search(
                data=[vec], anns_field="vector", param=search_params,
                limit=opt.topk, output_fields=["label"]
            )
            retrieved = results[0]

            # 3. ĐÁNH GIÁ RP VÀ CC
            correct_rp_1 = correct_rp_5 = 0
            correct_cc_1 = correct_cc_5 = 0

            for rank, r in enumerate(retrieved):
                retrieved_class = int(r.entity.get("label"))
                retrieved_group = CLASS_TO_GROUP.get(retrieved_class)

                if retrieved_group == gt_group:
                    correct_rp_5 += 1
                    if rank == 0: correct_rp_1 = 1

                if retrieved_class == gt_class:
                    correct_cc_5 += 1
                    if rank == 0: correct_cc_1 = 1

            # Lưu RP5 chung và RP5 riêng cho Group đó
            rp5_val = correct_rp_5 / opt.topk
            rp1_list.append(correct_rp_1)
            rp5_list.append(rp5_val)
            if gt_group in rp5_per_group:
                rp5_per_group[gt_group].append(rp5_val)

            # Lưu CC5 chung và CC5 riêng cho Class đó (chỉ PaL, PlL)
            if gt_group in TARGET_CC_GROUPS:
                cc5_val = correct_cc_5 / opt.topk
                cc1_list.append(correct_cc_1)
                cc5_list.append(cc5_val)
                if gt_class in cc5_per_class:
                    cc5_per_class[gt_class].append(cc5_val)

    # =============================
    # IN KẾT QUẢ THEO FORMAT PAPER
    # =============================
    print("\n" + "="*40)
    print("FINAL RESULTS (OVERALL)")
    print("="*40)
    print(f"Total valid queries: {len(rp1_list)}")
    print(f"Overall RP@5: {np.mean(rp5_list):.4f}")
    if len(cc5_list) > 0:
        print(f"Overall CC@5: {np.mean(cc5_list):.4f}")

    print("\n" + "="*40)
    print("RETRIEVAL PRECISION (RP5) PER GROUP")
    print("="*40)
    for g in ALL_GROUPS:
        vals = rp5_per_group[g]
        mean_val = np.mean(vals) if len(vals) > 0 else 0.0
        print(f"{g:<5} | RP5: {mean_val:.4f} | (Queries: {len(vals)})")

    print("\n" + "="*40)
    print("CLASS CONSISTENCY (CC5) PER CLASS")
    print("="*40)
    # Lấy thứ tự in giống y hệt biểu đồ trong paper
    ordered_cc_classes = [1, 2, 4, 5, 6, 7, 8, 13, 10, 11] 
    for c_id in ordered_cc_classes:
        abbr = CLASS_ID_TO_ABBR[c_id]
        vals = cc5_per_class[c_id]
        mean_val = np.mean(vals) if len(vals) > 0 else 0.0
        print(f"{abbr:<5} | CC5: {mean_val:.4f} | (Queries: {len(vals)})")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# import os
# import cv2
# import torch
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# from pymilvus import connections, Collection
# import argparse
# from collections import Counter

# import yolo_inf
# from run_single import load_model, pad_to_square_and_rgb
# from kaggle_secrets import UserSecretsClient


# # =============================
# # Hook
# # =============================

# def register_hooks(model, features_dict):

#     def get_hook(name):
#         def hook(module, input, output):
#             features_dict[name] = output
#         return hook

#     model.model[17].register_forward_hook(get_hook("small"))
#     model.model[20].register_forward_hook(get_hook("medium"))
#     model.model[23].register_forward_hook(get_hook("large"))


# # =============================
# # Main
# # =============================

# def main():

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--weights", required=True)
#     parser.add_argument("--image_dir", required=True)
#     parser.add_argument("--annotation_csv", required=True)
#     parser.add_argument("--img_size", type=int, default=384)
#     parser.add_argument("--conf", type=float, default=0.1)
#     parser.add_argument("--iou", type=float, default=0.4)
#     parser.add_argument("--metric", type=str, default="IP")
#     parser.add_argument("--topk", type=int, default=10)
#     opt = parser.parse_args()

#     # =============================
#     # Milvus connection
#     # =============================
#     user_secrets = UserSecretsClient()
#     MILVUS_URI = user_secrets.get_secret("MILVUS_URI")
#     MILVUS_TOKEN = user_secrets.get_secret("MILVUS_TOKEN")
#     COLLECTION_NAME = user_secrets.get_secret("COLLECTION_NAME")

#     print("Connecting to Milvus...")
#     connections.connect(uri=MILVUS_URI, token=MILVUS_TOKEN)
#     collection = Collection(COLLECTION_NAME)
#     collection.load()

#     # =============================
#     # Load library label distribution
#     # =============================
#     print("Loading library label distribution...")
#     all_entities = collection.query(
#         expr="label >= 0",
#         output_fields=["label"]
#     )

#     label_counter = Counter([r["label"] for r in all_entities])

#     # =============================
#     # Load model
#     # =============================
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = load_model(opt.weights, device)

#     features = {}
#     register_hooks(model, features)

#     stride_map = {0: 8, 1: 16, 2: 32}
#     branch_name_map = {0: "small", 1: "medium", 2: "large"}

#     # =============================
#     # Load test annotation
#     # =============================
#     df = pd.read_csv(opt.annotation_csv)
#     image_list = df["image_id"].unique().tolist()

#     search_params = {
#         "metric_type": opt.metric,
#         "params": {"nprobe": 10}
#     }

#     K_list = [1, 5, 10]
#     mp_at_k = {k: [] for k in K_list}
#     r_at_k = {k: [] for k in K_list}
#     ap_list = []

#     print("Starting evaluation...")

#     for img_name in tqdm(image_list):

#         img_path = os.path.join(opt.image_dir, img_name)
#         if not os.path.exists(img_path):
#             img_path = os.path.join(opt.image_dir, img_name + ".png")

#         img = cv2.imread(img_path)
#         if img is None:
#             continue

#         padded = pad_to_square_and_rgb(img)

#         boxes, scores, labels, branch_ids, boxes_resized = yolo_inf.detect1Image(
#             padded, opt.img_size, model, device, opt.conf, opt.iou
#         )

#         for i, box in enumerate(boxes_resized):

#             branch_id = int(branch_ids[i])
#             branch_name = branch_name_map[branch_id]
#             stride = stride_map[branch_id]

#             x1, y1, x2, y2 = box
#             cx = (x1 + x2) / 2.0
#             cy = (y1 + y2) / 2.0

#             gx = int(cx / stride)
#             gy = int(cy / stride)

#             fmap = features[branch_name][0]
#             C, H, W = fmap.shape

#             gx = np.clip(gx, 0, W - 1)
#             gy = np.clip(gy, 0, H - 1)

#             vec = fmap[:, gy, gx]
#             vec = torch.nn.functional.normalize(vec, dim=0)
#             vec = vec.detach().cpu().numpy()

#             # =============================
#             # SEARCH
#             # =============================
#             results = collection.search(
#                 data=[vec],
#                 anns_field="vector",
#                 param=search_params,
#                 limit=opt.topk + 1,   # +1 để loại self-match
#                 output_fields=["label", "image_id"]
#             )

#             retrieved = results[0]

#             query_label = int(labels[i])

#             # Remove self-match
#             retrieved_filtered = [
#                 r for r in retrieved
#                 if r.entity.get("image_id") != img_name
#             ]

#             retrieved_labels = [
#                 r.entity.get("label") for r in retrieved_filtered
#             ]

#             total_positives = label_counter[query_label]

#             if total_positives <= 1:
#                 continue

#             # =============================
#             # Precision & Recall
#             # =============================
#             for K in K_list:

#                 topk = retrieved_labels[:K]
#                 correct = sum(1 for l in topk if l == query_label)

#                 precision = correct / K
#                 recall = correct / total_positives

#                 mp_at_k[K].append(precision)
#                 r_at_k[K].append(recall)

#             # =============================
#             # Average Precision
#             # =============================
#             correct_so_far = 0
#             precision_accum = []

#             for rank, l in enumerate(retrieved_labels):

#                 if l == query_label:
#                     correct_so_far += 1
#                     precision_accum.append(correct_so_far / (rank + 1))

#             if len(precision_accum) > 0:
#                 ap = sum(precision_accum) / total_positives
#             else:
#                 ap = 0

#             ap_list.append(ap)

#     # =============================
#     # FINAL RESULTS
#     # =============================
#     print("\n===== FINAL RETRIEVAL RESULTS =====")

#     for K in K_list:
#         print(f"mP@{K}: {np.mean(mp_at_k[K]) * 100:.2f}")
#         print(f"R@{K}: {np.mean(r_at_k[K]) * 100:.2f}")

#     print(f"mAP: {np.mean(ap_list) * 100:.2f}")


# if __name__ == "__main__":
#     main()