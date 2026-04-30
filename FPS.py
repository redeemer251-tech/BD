import cv2
import torch
import time
import os
from ultralytics import YOLO

WEIGHTS    = 'runs/detect/runs/train/bce_run/weights/best.pt'
VIDEO_IN   = 'test/RDT_20260330_234414.mp4'
VIDEO_OUT  = 'test/RDT_20260330_234414_output_annotated.mp4'
CONF_THRES = 0.25
IOU_THRES  = 0.45
IMG_SIZE   = 640
CLASSES    = ['helicopter', 'airplane', 'uav']

CLASS_COLORS = {0: (0, 0, 255), 1: (0, 255, 0), 2: (255, 100, 0)}

def run_inference_on_video(weights, video_in, video_out):
    model = YOLO(weights)
    model.model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    cap = cv2.VideoCapture(video_in)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_in}")

    fps        = cap.get(cv2.CAP_PROP_FPS)
    width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total / fps if fps > 0 else 0

    print(f"Input:    {video_in}")
    print(f"Size:     {width}x{height}  |  FPS: {fps:.1f}  |  Frames: {total}  |  Duration: {duration_s:.1f}s")

    os.makedirs(os.path.dirname(video_out) if os.path.dirname(video_out) else '.', exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_out, fourcc, fps, (width, height))

    print("Warming up...")
    for _ in range(10):
        dummy = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE).to(device)
        model.predict(source=dummy, verbose=False)

    frame_times = []
    frame_idx   = 0
    total_dets  = 0

    print("Running inference...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if device == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        results = model.predict(
            source=frame,
            imgsz=IMG_SIZE,
            conf=CONF_THRES,
            iou=IOU_THRES,
            verbose=False,
            device=device,
        )

        if device == 'cuda':
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        frame_times.append(t1 - t0)

        annotated = frame.copy()
        result = results[0]

        if result.boxes is not None and len(result.boxes):
            total_dets += len(result.boxes)
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id = int(box.cls[0].item())
                conf   = float(box.conf[0].item())
                color  = CLASS_COLORS.get(cls_id, (255, 255, 255))
                label  = f"{CLASSES[cls_id]} {conf:.2f}"

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(annotated, (x1, y1 - lh - baseline - 4), (x1 + lw, y1), color, -1)

                cv2.putText(annotated, label, (x1, y1 - baseline - 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        live_fps = 1.0 / frame_times[-1] if frame_times else 0
        cv2.putText(annotated, f"Frame {frame_idx+1}/{total}  |  {live_fps:.1f} FPS",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        writer.write(annotated)
        frame_idx += 1

        if frame_idx % 50 == 0:
            elapsed   = sum(frame_times)
            eta       = (elapsed / frame_idx) * (total - frame_idx)
            avg_fps   = frame_idx / elapsed
            print(f"  Frame {frame_idx}/{total}  |  avg {avg_fps:.1f} FPS  |  ETA {eta:.0f}s")

    cap.release()
    writer.release()

    if frame_times:
        avg_ms  = (sum(frame_times) / len(frame_times)) * 1000
        avg_fps = 1000 / avg_ms
        min_fps = 1000 / (max(frame_times) * 1000)
        max_fps = 1000 / (min(frame_times) * 1000)

        print("\n─── Rezultāti ───────────────────────────────────────")
        print(f"Output saved:      {video_out}")
        print(f"Frames processed:  {frame_idx}")
        print(f"Total detections:  {total_dets}")
        print(f"Avg inference:     {avg_ms:.2f} ms/frame")
        print(f"Avg FPS:           {avg_fps:.1f}")
        print(f"Min FPS:           {min_fps:.1f}  (slowest frame)")
        print(f"Max FPS:           {max_fps:.1f}  (fastest frame)")
        print("────────────────────────────────────────────────────")

    return {
        'frames':      frame_idx,
        'detections':  total_dets,
        'avg_fps':     avg_fps,
        'avg_ms':      avg_ms,
    }


if __name__ == '__main__':
    metrics = run_inference_on_video(WEIGHTS, VIDEO_IN, VIDEO_OUT)