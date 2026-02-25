#!/usr/bin/env python3
import os
import sys
import argparse
import cv2
import torch
import numpy as np

CUR_PATH = os.path.dirname(os.path.realpath(__file__)) + '/'
sys.path.append(CUR_PATH + 'yolo5')

import yolo_inf


def load_model(weights, device):
    ckpt = torch.load(weights, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        model = ckpt['model'].float()
    else:
        # fallback: try loading with YOLOv5 loader
        from models.experimental import attempt_load
        model = attempt_load(weights, map_location=device)
    model.to(device).eval()
    return model


def pad_to_square_and_rgb(img):
    # img: BGR from cv2
    im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = im_rgb.shape[:2]
    if h != w:
        new_size = max(h, w)
        result = np.zeros((new_size, new_size, 3), dtype=im_rgb.dtype)
        result[0:h, 0:w, :] = im_rgb[0:h, 0:w, :]
        return result
    return im_rgb


def draw_boxes_and_save(im_bgr, boxes, scores, labels, names, out_path):
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        conf = scores[i] if i < len(scores) else 0
        cls = labels[i] if i < len(labels) else 0
        cv2.rectangle(im_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(im_bgr, f"{names[int(cls)]} {conf:.2f}", (x1, max(0, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    cv2.imwrite(out_path, im_bgr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, required=True, help='path to .pt checkpoint (torch.save)')
    parser.add_argument('--images', nargs='+', required=True, help='one or more image paths')
    parser.add_argument('--img-size', type=int, default=640)
    parser.add_argument('--conf-thres', type=float, default=0.01)
    parser.add_argument('--iou-thres', type=float, default=0.4)
    parser.add_argument('--out', type=str, default='out_single', help='output folder')
    opt = parser.parse_args()

    os.makedirs(opt.out, exist_ok=True)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = load_model(opt.weights, device)

    features = {}

    def get_hook(name):
        def hook(module, input, output):
            features[name] = output
        return hook

    # Based on your printed architecture:
    model.model[17].register_forward_hook(get_hook("small"))
    model.model[20].register_forward_hook(get_hook("medium"))
    model.model[23].register_forward_hook(get_hook("large"))


    # try to get class names from model (if available)
    try:
        names = model.module.names if hasattr(model, 'module') else model.names
    except Exception:
        names = [str(i) for i in range(100)]

    stride_map = {
        0: 8,   # small
        1: 16,  # medium
        2: 32   # large
    }

    branch_name_map = {
        0: "small",
        1: "medium",
        2: "large"
    }

    for img_path in opt.images:
        print('Processing', img_path)
        img = cv2.imread(img_path)
        if img is None:
            print('  Cannot read image, skipping')
            continue

        padded = pad_to_square_and_rgb(img)
        boxes, scores, labels, branch, boxes_resized = yolo_inf.detect1Image(padded, opt.img_size, model, device, opt.conf_thres, opt.iou_thres)
        for k in features:
            print(f"{k} feature shape:", features[k].shape)

        print(f'  Found {len(boxes)} boxes')
        for b, s, l in zip(boxes, scores, labels):
            print('   ', int(l), f'{s:.3f}', list(map(int, b)))

        lesion_features = []

        for i, box in enumerate(boxes_resized):
            x1, y1, x2, y2 = box
            branch_id = int(branch[i])

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            stride = stride_map[branch_id]
            branch_name = branch_name_map[branch_id]

            # grid cell đúng chuẩn YOLO
            gx = int(cx / stride)
            gy = int(cy / stride)

            fmap = features[branch_name][0]  # remove batch dim
            C, H, W = fmap.shape

            gx = np.clip(gx, 0, W - 1)
            gy = np.clip(gy, 0, H - 1)

            vec = fmap[:, gy, gx]

            # Normalize (rất quan trọng cho retrieval)
            vec = torch.nn.functional.normalize(vec, dim=0)

            lesion_features.append({
                "bbox": box,
                "branch_id": branch_id,
                "branch_name": branch_name,
                "feature": vec.detach().cpu()
            })

            print(f"  Feature dim ({branch_name}):", vec.shape)
        print(lesion_features)
        out_file = os.path.join(opt.out, os.path.basename(img_path))
        # save visualization on padded image (quick debug)
        padded_bgr = cv2.cvtColor(padded, cv2.COLOR_RGB2BGR)
        draw_boxes_and_save(padded_bgr, boxes, scores, labels, names, out_file)


if __name__ == '__main__':
    main()
