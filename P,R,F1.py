import torch
from ultralytics import YOLO

# ── CONFIGURATION ──────────────────────────────────────────
WEIGHTS_PATH = 'runs/detect/runs/train/bce_run/weights/best.pt'
DATA_YAML    = 'dataset_final/data.yaml' # Path to your dataset config
IMG_SIZE     = 640
# ───────────────────────────────────────────────────────────

def evaluate_model(weights, data_yaml):
    # Load the model
    model = YOLO(weights)
    
    print(f"Starting validation on {data_yaml}...")
    
    # Run validation (equivalent to the logic in PRECALLF1.py)
    # This automatically processes the 'test' split defined in your YAML
    results = model.val(
        data=data_yaml,
        split='test',
        imgsz=IMG_SIZE,
        conf=0.25,
        iou=0.5,
        verbose=True
    )

    # Extracting metrics
    precision = results.box.mp  # Mean Precision
    recall    = results.box.mr     # Mean Recall
    
    # Calculate F1-score
    f1 = 2 * (precision * recall) / (precision + recall + 1e-9)

    print("\n─── Final Metrics ────────────────────────────────")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print("──────────────────────────────────────────────────")

if __name__ == '__main__':
    evaluate_model(WEIGHTS_PATH, DATA_YAML)