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
    ckpt = torch.load(weights, map_location=device)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        model = ckpt['model'].float()
    else:
        # fallback: try loading with YOLOv5 loader
        from models.experimental import attempt_load
        model = attempt_load(weights, map_location=device, weights_only=False)
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

    # try to get class names from model (if available)
    try:
        names = model.module.names if hasattr(model, 'module') else model.names
    except Exception:
        names = [str(i) for i in range(100)]

    for img_path in opt.images:
        print('Processing', img_path)
        img = cv2.imread(img_path)
        if img is None:
            print('  Cannot read image, skipping')
            continue

        padded = pad_to_square_and_rgb(img)
        boxes, scores, labels = yolo_inf.detect1Image(padded, opt.img_size, model, device, opt.conf_thres, opt.iou_thres)

        print(f'  Found {len(boxes)} boxes')
        for b, s, l in zip(boxes, scores, labels):
            print('   ', int(l), f'{s:.3f}', list(map(int, b)))

        out_file = os.path.join(opt.out, os.path.basename(img_path))
        # save visualization on padded image (quick debug)
        padded_bgr = cv2.cvtColor(padded, cv2.COLOR_RGB2BGR)
        draw_boxes_and_save(padded_bgr, boxes, scores, labels, names, out_file)


if __name__ == '__main__':
    main()
