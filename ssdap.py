import os
import time
import json
import torch
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import ssd300_vgg16, SSD300_VGG16_Weights
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.ops import box_iou
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

DATASET_DIR = r'C:\Users\Admin\Desktop\BAKAULARA DARBS\dataset_final'
OUTPUT_DIR  = r'C:\Users\Admin\Desktop\BAKAULARA DARBS\runs\ssd'
CLASS_NAMES = ['__background__', 'helicopter', 'airplane', 'uav']
NUM_CLASSES = 4
BATCH_SIZE  = 8
BASE_LR     = 0.0005
MAX_TIME_SEC = 7560
LR_STEPS    = (30000, 36000)
CONF_THRES  = 0.25
WARMUP_ITERS = 500

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class COCODataset(Dataset):
    def __init__(self, images_dir, annotations_json, img_size=300):
        self.images_dir = Path(images_dir)
        self.img_size   = img_size
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
        if img is None:
            print(f"  Warning: could not read image {img_info['file_name']}, returning blank.")
            img = np.zeros((300, 300, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        img_resized = cv2.resize(img, (self.img_size, self.img_size))
        img_tensor  = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0

        anns   = self.annotations.get(img_id, [])
        boxes, labels = [], []
        for ann in anns:
            bx, by, bw, bh = ann['bbox']
            if bw > 0 and bh > 0:
                x1 = (bx / w)        * self.img_size
                y1 = (by / h)        * self.img_size
                x2 = ((bx+bw) / w)  * self.img_size
                y2 = ((by+bh) / h)  * self.img_size
                x1 = max(0.0, min(x1, self.img_size - 1))
                y1 = max(0.0, min(y1, self.img_size - 1))
                x2 = max(x1 + 1.0, min(x2, float(self.img_size)))
                y2 = max(y1 + 1.0, min(y2, float(self.img_size)))
                label = ann['category_id'] + 1
                if label < 1 or label >= NUM_CLASSES:
                    print(f"  Warning: skipping ann with category_id={ann['category_id']} "
                          f"(out of range for NUM_CLASSES={NUM_CLASSES})")
                    continue
                boxes.append([x1, y1, x2, y2])
                labels.append(label)

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

def train(model, train_loader, optimizer, scheduler, max_time_sec, warmup_iters=500):
    model.train()
    start_time  = time.time()
    total_iters = 0

    print(f"Training SSD for exactly {max_time_sec / 3600:.2f} hours...")
    print(f"  LR warmup over first {warmup_iters} iterations, then MultiStepLR kicks in.")
    
    keep_training = True 
    
    while keep_training:
        for images, targets in train_loader:
            elapsed = time.time() - start_time
            
            if elapsed >= max_time_sec:
                keep_training = False
                break

            images  = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            if total_iters < warmup_iters:
                warmup_factor = (total_iters + 1) / warmup_iters
                for pg in optimizer.param_groups:
                    pg['lr'] = BASE_LR * warmup_factor

            loss_dict = model(images, targets)
            losses    = sum(loss_dict.values())


            if not torch.isfinite(losses):
                print(f"  Warning: non-finite loss ({losses.item()}) at iter {total_iters+1}, skipping batch.")
                optimizer.zero_grad()
                continue

            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            scheduler.step()

            total_iters += 1
            if total_iters % 500 == 0:
                elapsed = time.time() - start_time
                eta     = max_time_sec - elapsed
                print(f"  Iter {total_iters}  "
                      f"loss: {losses.item():.4f}  "
                      f"elapsed: {elapsed/3600:.2f}h  "
                      f"ETA: {eta/3600:.2f}h")

    return time.time() - start_time

def evaluate(model, dataset_dir, split, conf_thres=0.25):
    model.eval()
    gt_json  = f'{dataset_dir}/annotations_{split}.json'
    imgs_dir = f'{dataset_dir}/images/{split}'
    coco_gt  = COCO(gt_json)
    coco_preds = []

    with torch.no_grad():
        for img_id in coco_gt.getImgIds():
            img_info = coco_gt.loadImgs(img_id)[0]
            img = cv2.imread(os.path.join(imgs_dir, img_info['file_name']))
            if img is None:
                continue
            h, w = img.shape[:2]
            img_rgb     = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (300, 300))
            img_tensor  = torch.from_numpy(img_resized).permute(2,0,1).float() / 255.0
            img_tensor  = img_tensor.to(device)

            outputs = model([img_tensor])[0]

            for box, label, score in zip(
                    outputs['boxes'], outputs['labels'], outputs['scores']):
                if score < conf_thres:
                    continue
                x1, y1, x2, y2 = box.tolist()
                x1 = x1 / 300 * w;  x2 = x2 / 300 * w
                y1 = y1 / 300 * h;  y2 = y2 / 300 * h
                coco_preds.append({
                    'image_id':    img_id,
                    'category_id': int(label.item()) - 1,
                    'bbox':        [x1, y1, x2-x1, y2-y1],
                    'score':       float(score.item()),
                })

    if not coco_preds:
        print(f"  Warning: no predictions above threshold for {split}")
        return {'precision': 0, 'recall': 0, 'f1': 0,
                'mAP_50': 0, 'mAP_50_95': 0}

    coco_dt   = coco_gt.loadRes(coco_preds)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    map_50    = float(coco_eval.stats[1])
    map_50_95 = float(coco_eval.stats[0])

    gt_by_img   = {}
    for ann in coco_gt.loadAnns(coco_gt.getAnnIds()):
        x, y, bw, bh = ann['bbox']
        gt_by_img.setdefault(ann['image_id'], []).append(
            [x, y, x+bw, y+bh, ann['category_id']])

    pred_by_img = {}
    for p in coco_preds:
        x, y, bw, bh = p['bbox']
        pred_by_img.setdefault(p['image_id'], []).append(
            [x, y, x+bw, y+bh, p['category_id'], p['score']])

    TP = FP = FN = 0
    for img_id in coco_gt.getImgIds():
        gts   = gt_by_img.get(img_id, [])
        preds = pred_by_img.get(img_id, [])
        if not gts and not preds:
            continue
        if not preds:
            FN += len(gts); continue
        if not gts:
            FP += len(preds); continue

        gt_boxes   = torch.tensor([[g[0],g[1],g[2],g[3]] for g in gts],   dtype=torch.float32)
        pred_boxes = torch.tensor([[p[0],p[1],p[2],p[3]] for p in preds], dtype=torch.float32)
        iou_matrix = box_iou(pred_boxes, gt_boxes)

        matched_gt = set()
        for pi in range(len(preds)):
            best_iou, best_gi = iou_matrix[pi].max(0)
            best_gi = best_gi.item()
            if best_iou >= 0.5 and best_gi not in matched_gt:
                if preds[pi][4] == gts[best_gi][4]:
                    TP += 1
                    matched_gt.add(best_gi)
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
    }

if __name__ == '__main__':
    weights = SSD300_VGG16_Weights.DEFAULT
    model   = ssd300_vgg16(weights=weights)
    in_channels = [512, 1024, 512, 256, 256, 256]
    num_anchors = model.anchor_generator.num_anchors_per_location()
    model.head.classification_head = SSDClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=NUM_CLASSES,
    )
    model.to(device)

    train_dataset = COCODataset(
        f'{DATASET_DIR}/images/train',
        f'{DATASET_DIR}/annotations_train.json',
        img_size=300,
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

    training_time = train(model, train_loader, optimizer, scheduler, MAX_TIME_SEC, WARMUP_ITERS)
    h = int(training_time // 3600)
    m = int((training_time % 3600) // 60)
    s = int(training_time % 60)
    print(f"\nTraining complete: {h}h {m}m {s}s")

    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'model_final.pth'))

    conditions = ['test', 'test_fog', 'test_rain', 'test_lowlight']
    rows = []
    for cond in conditions:
        label = cond.replace('test_', '').replace('test', 'clear')
        print(f"\n  Evaluating [{label}]...")
        metrics = evaluate(model, DATASET_DIR, cond, CONF_THRES)
        rows.append({'model': 'SSD', 'condition': label, **metrics,
                     'training_time': f'{h}h {m}m {s}s'})
        print(f"    P: {metrics['precision']}  R: {metrics['recall']}  "
              f"F1: {metrics['f1']}  mAP@0.5: {metrics['mAP_50']}")

    df = pd.DataFrame(rows)
    out_csv = os.path.join(OUTPUT_DIR, 'results_ssd.csv')
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")