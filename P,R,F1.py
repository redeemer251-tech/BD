import torch
from ultralytics import YOLO

WEIGHTS_PATH = 'runs/detect/runs/train/bce_run/weights/best.pt'
DATA_YAML    = 'dataset_final/data.yaml'
IMG_SIZE     = 640

def evaluate_model(weights, data_yaml):
    model = YOLO(weights)
    
    print(f"Starting validation on {data_yaml}...")
    
    results = model.val(
        data=data_yaml,
        split='test',
        imgsz=IMG_SIZE,
        conf=0.25,
        iou=0.5,
        verbose=True
    )

    precision = results.box.mp
    recall    = results.box.mr
    
    f1 = 2 * (precision * recall) / (precision + recall + 1e-9)

    print("\n─── Gala rezultāti ────────────────────────────────")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("──────────────────────────────────────────────────")

if __name__ == '__main__':
    evaluate_model(WEIGHTS_PATH, DATA_YAML)