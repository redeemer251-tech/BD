import os
import time
import json
import torch
import cv2
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

DATASET_DIR = r'C:\Users\Admin\Desktop\BAKAULARA DARBS\dataset_final'
OUTPUT_DIR  = r'C:\Users\Admin\Desktop\BAKAULARA DARBS\runs\faster_rcnn'
CLASS_NAMES = ['__background__', 'helicopter', 'airplane', 'uav']
NUM_CLASSES = 4
BATCH_SIZE  = 8
BASE_LR     = 0.005
MAX_ITER    = 40000
TRAIN_DURATION = 2 * 3600 + 6 * 60
LR_STEPS    = (22000, 27000)
CONF_THRES  = 0.25

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class COCODataset(Dataset):
    def __init__(self, images_dir, annotations_json):
        self.images_dir = Path(images_dir)
        with open(annotations_json) as f:
            data = json.load(f)
        self.images      = {img['id']: img for img in data['images']}
        self.img_ids     = [img['id'] for img in data['images']]
        self.annotations = {}
        for ann in data['annotations']:
            self.annotations.setdefault(ann['image_id'], []).append(ann)

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id   = self.img_ids[idx]
        img_info = self.images[img_id]
        img = cv2.imread(str(self.images_dir / img_info['file_name']))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        anns   = self.annotations.get(img_id, [])
        boxes, labels = [], []
        for ann in anns:
            x, y, w, h = ann['bbox']
            if w > 0 and h > 0:
                boxes.append([x, y, x + w, y + h])
                labels.append(ann['category_id'] + 1)

        target = {
            'boxes':    torch.tensor(boxes,  dtype=torch.float32) if boxes
                        else torch.zeros((0, 4), dtype=torch.float32),
            'labels':   torch.tensor(labels, dtype=torch.int64)   if labels
                        else torch.zeros((0,),  dtype=torch.int64),
            'image_id': torch.tensor([img_id]),
        }
        return img_tensor, target

def collate_fn(batch):
    return tuple(zip(*batch))

def train(model, train_loader, optimizer, scheduler, duration_seconds):
    model.train()
    start_time  = time.time()
    total_iters = 0
    epoch       = 0

    print(f"Training for {duration_seconds/3600:.2f}h (batch size {BATCH_SIZE})...")

    while (time.time() - start_time) < duration_seconds:
        epoch += 1
        for images, targets in train_loader:
            if (time.time() - start_time) >= duration_seconds:
                break

            images  = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses    = sum(loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            scheduler.step()

            total_iters += 1

            if total_iters % 500 == 0:
                elapsed = time.time() - start_time
                remaining = duration_seconds - elapsed
                print(f"  Iter {total_iters}  "
                      f"loss: {losses.item():.4f}  "
                      f"elapsed: {elapsed/3600:.2f}h  "
                      f"remaining: {remaining/60:.1f}min")

    return time.time() - start_time

def evaluate(model, dataset_dir, split, conf_thres=0.25):
    """
    Returns precision, recall, F1 and mAP metrics for a given test split.
    split: 'test', 'test_fog', 'test_rain', 'test_lowlight'
    """
    model.eval()
    gt_json   = f'{dataset_dir}/annotations_{split}.json'
    imgs_dir  = f'{dataset_dir}/images/{split}'
    coco_gt   = COCO(gt_json)
    img_ids   = coco_gt.getImgIds()
    coco_preds = []

    with torch.no_grad():
        for img_id in img_ids:
            img_info = coco_gt.loadImgs(img_id)[0]
            img_path = os.path.join(imgs_dir, img_info['file_name'])
            img = cv2.imread(img_path)
            if img is None:
                continue
            img_rgb    = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img_rgb).permute(2,0,1).float() / 255.0
            img_tensor = img_tensor.to(device)

            outputs = model([img_tensor])[0]

            for box, label, score in zip(
                    outputs['boxes'], outputs['labels'], outputs['scores']):
                if score < conf_thres:
                    continue
                x1, y1, x2, y2 = box.tolist()
                coco_preds.append({
                    'image_id':    img_id,
                    'category_id': int(label.item()) - 1,
                    'bbox':        [x1, y1, x2 - x1, y2 - y1],
                    'score':       float(score.item()),
                })

    if not coco_preds:
        print(f"  Warning: no predictions above threshold for {split}")
        return {'precision': 0, 'recall': 0, 'f1': 0,
                'mAP_50': 0, 'mAP_50_95': 0}

    coco_dt   = coco_gt.loadRes(coco_preds)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.params.iouThrs = np.linspace(0.5, 0.95, 10)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    map_50    = float(coco_eval.stats[1])
    map_50_95 = float(coco_eval.stats[0])

    from torchvision.ops import box_iou

    gt_by_img  = {}
    for ann in coco_gt.loadAnns(coco_gt.getAnnIds()):
        x, y, w, h = ann['bbox']
        gt_by_img.setdefault(ann['image_id'], []).append(
            [x, y, x+w, y+h, ann['category_id']])

    pred_by_img = {}
    for p in coco_preds:
        x, y, w, h = p['bbox']
        pred_by_img.setdefault(p['image_id'], []).append(
            [x, y, x+w, y+h, p['category_id'], p['score']])

    TP = FP = FN = 0
    for img_id in img_ids:
        gts   = gt_by_img.get(img_id, [])
        preds = pred_by_img.get(img_id, [])

        if not gts and not preds:
            continue
        if not preds:
            FN += len(gts)
            continue
        if not gts:
            FP += len(preds)
            continue

        gt_boxes   = torch.tensor([[g[0],g[1],g[2],g[3]] for g in gts],   dtype=torch.float32)
        pred_boxes = torch.tensor([[p[0],p[1],p[2],p[3]] for p in preds], dtype=torch.float32)
        iou_matrix = box_iou(pred_boxes, gt_boxes)  # shape: [n_preds, n_gts]

        matched_gt   = set()
        matched_pred = set()
        for pi in range(len(preds)):
            best_iou, best_gi = iou_matrix[pi].max(0)
            best_gi = best_gi.item()
            if best_iou >= 0.5 and best_gi not in matched_gt:
                if preds[pi][4] == gts[best_gi][4]:
                    TP += 1
                    matched_gt.add(best_gi)
                    matched_pred.add(pi)
                else:
                    FP += 1
            else:
                FP += 1

        FN += len(gts) - len(matched_gt)

    precision = TP / (TP + FP + 1e-9)
    recall    = TP / (TP + FN + 1e-9)
    f1        = 2 * precision * recall / (precision + recall + 1e-9)

    return {
        'precision': round(precision, 4),
        'recall':    round(recall,    4),
        'f1':        round(f1,        4),
        'mAP_50':    round(map_50,    4),
        'mAP_50_95': round(map_50_95, 4),
        'TP': TP, 'FP': FP, 'FN': FN,
    }

if __name__ == '__main__':
    import pandas as pd

    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
    model.to(device)

    train_dataset = COCODataset(
        f'{DATASET_DIR}/images/train',
        f'{DATASET_DIR}/annotations_train.json',
    )
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, collate_fn=collate_fn,
    )

    params    = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=BASE_LR,
                                momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=list(LR_STEPS), gamma=0.1)

    training_time = train(model, train_loader, optimizer, scheduler, TRAIN_DURATION)
    h = int(training_time // 3600)
    m = int((training_time % 3600) // 60)
    s = int(training_time % 60)
    print(f"\nTraining complete: {h}h {m}m {s}s")

    weights_path = os.path.join(OUTPUT_DIR, 'model_final.pth')
    torch.save(model.state_dict(), weights_path)
    print(f"Saved: {weights_path}")

    print("\nEvaluating on all test conditions...")
    conditions = ['test', 'test_fog', 'test_rain', 'test_lowlight']
    rows = []
    for cond in conditions:
        label = cond.replace('test_', '').replace('test', 'clear')
        print(f"\n  [{label}]")
        m = evaluate(model, DATASET_DIR, cond, CONF_THRES)
        rows.append({'model': 'Faster R-CNN', 'condition': label, **m,
                     'training_time': f'{h}h {m_}m {s}s',
                     'training_seconds': round(training_time)})

    df = pd.DataFrame(rows)
    out_csv = os.path.join(OUTPUT_DIR, 'results_faster_rcnn.csv')
    print("\n", df[['condition','precision','recall','f1',
                     'mAP_50','mAP_50_95']].to_string(index=False))
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")