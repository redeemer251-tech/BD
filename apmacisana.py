from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8s.pt')   # Downloads pretrained weights automatically
    results = model.train(
    data='dataset_final/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    lr0=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    patience=20,             # Early stopping: stops if no improvement for 20 epochs
    project='runs/train',
    name='bce_run',
    seed=42                  # Fix seed for reproducibility
)