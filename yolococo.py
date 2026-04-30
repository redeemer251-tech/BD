# convert_to_coco.py
import os
import json
import cv2
from pathlib import Path

DATASET_DIR = r'C:/Users/Admin/Desktop/BAKAULARA DARBS/dataset_final'
CLASS_NAMES = ['helicopter', 'airplane', 'uav']   # must match your data.yaml

def yolo_to_coco(images_dir, labels_dir, output_json):
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    coco = {
        'info':        {'description': 'Aircraft Detection Dataset'},
        'categories':  [{'id': i, 'name': n, 'supercategory': 'aircraft'}
                        for i, n in enumerate(CLASS_NAMES)],
        'images':      [],
        'annotations': [],
    }

    ann_id = 0
    img_id = 0

    for img_path in sorted(images_dir.glob('*')):
        if img_path.suffix.lower() not in ['.jpg', '.jpeg', '.png']:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        coco['images'].append({
            'id':        img_id,
            'file_name': img_path.name,
            'height':    h,
            'width':     w,
        })

        lbl_path = labels_dir / (img_path.stem + '.txt')
        if lbl_path.exists():
            for line in lbl_path.read_text().strip().splitlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id    = int(parts[0])
                xc, yc, bw, bh = map(float, parts[1:])

                # Convert normalized YOLO → absolute COCO (x_min, y_min, w, h)
                abs_w  = bw * w
                abs_h  = bh * h
                x_min  = (xc - bw / 2) * w
                y_min  = (yc - bh / 2) * h

                coco['annotations'].append({
                    'id':           ann_id,
                    'image_id':     img_id,
                    'category_id':  cls_id,
                    'bbox':         [round(x_min, 2), round(y_min, 2),
                                     round(abs_w, 2), round(abs_h, 2)],
                    'area':         round(abs_w * abs_h, 2),
                    'iscrowd':      0,
                })
                ann_id += 1

        img_id += 1

    with open(output_json, 'w') as f:
        json.dump(coco, f)
    print(f"Saved {img_id} images, {ann_id} annotations → {output_json}")


for split in ['train', 'val', 'test']:
    yolo_to_coco(
        images_dir=f'{DATASET_DIR}/images/{split}',
        labels_dir=f'{DATASET_DIR}/labels/{split}',
        output_json=f'{DATASET_DIR}/annotations_{split}.json',
    )

# Also generate COCO JSONs for weather conditions (same labels, augmented images)
for cond in ['fog', 'rain', 'lowlight']:
    yolo_to_coco(
        images_dir=f'{DATASET_DIR}/images/test_{cond}',
        labels_dir=f'{DATASET_DIR}/labels/test',   # same labels as clear test
        output_json=f'{DATASET_DIR}/annotations_test_{cond}.json',
    )