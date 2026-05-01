import cv2
import torch
import time
import os
import numpy as np
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator

WEIGHTS    = r'C:\Users\Admin\Desktop\BAKAULARA DARBS\runs\ssd\model_final.pth'
VIDEO_IN   = r'test\test_20190925_124000_1_9_visible.mp4'
VIDEO_OUT  = r'test\test_20190925_124000_1_9_visible_output_ssd.mp4'
CONF_THRES = 0.25
NUM_CLASSES = 4
CLASSES    = ['__bg__', 'helicopter', 'airplane', 'uav']
CLASS_COLORS = {1: (0, 255, 0), 2: (255, 100, 0), 3: (0, 0, 255)}


def load_model(weights_path, num_classes, device):
    """
    Loads ssd300_vgg16 and replaces the classification head to match
    the number of classes used during training.
    """
    model = ssd300_vgg16(weights=None)

    in_channels = [512, 1024, 512, 256, 256, 256]
    num_anchors = model.anchor_generator.num_anchors_per_location()
    model.head.classification_head = SSDClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes,
    )

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def frame_to_tensor(frame, device):
    """Convert a BGR OpenCV frame to a normalised float tensor on device."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return tensor.to(device)


def run_inference_on_video(weights, video_in, video_out):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = load_model(weights, NUM_CLASSES, device)
    print(f"Model loaded from: {weights}")

    cap = cv2.VideoCapture(video_in)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_in}")

    fps        = cap.get(cv2.CAP_PROP_FPS)
    width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total / fps if fps > 0 else 0

    print(f"\nInput:    {video_in}")
    print(f"Size:     {width}x{height}  |  FPS: {fps:.1f}  |  Frames: {total}  |  Duration: {duration_s:.1f}s")

    os.makedirs(os.path.dirname(video_out) if os.path.dirname(video_out) else '.', exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_out, fourcc, fps, (width, height))

    print("\nModeļa sagatavošana...")
    dummy_tensor = torch.zeros(1, 3, 300, 300).to(device)
    with torch.no_grad():
        for _ in range(10):
            model(list(dummy_tensor))
    if device.type == 'cuda':
        torch.cuda.synchronize()
    print("Warmup complete.")

    frame_times = []
    frame_idx   = 0
    total_dets  = 0

    print("\nRunning inference...")
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            tensor = frame_to_tensor(frame, device)

            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            outputs = model([tensor])[0]

            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            frame_times.append(t1 - t0)

            annotated = frame.copy()
            boxes  = outputs['boxes']
            labels = outputs['labels']
            scores = outputs['scores']

            keep = scores >= CONF_THRES
            boxes, labels, scores = boxes[keep], labels[keep], scores[keep]
            total_dets += len(boxes)

            for box, cls_id, conf in zip(boxes, labels, scores):
                x1, y1, x2, y2 = map(int, box.tolist())
                cls_id = int(cls_id.item())
                conf   = float(conf.item())
                color  = CLASS_COLORS.get(cls_id, (255, 255, 255))
                name   = CLASSES[cls_id] if cls_id < len(CLASSES) else str(cls_id)
                label_text = f"{name} {conf:.2f}"

                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                (lw, lh), baseline = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                cv2.rectangle(
                    annotated,
                    (x1, y1 - lh - baseline - 4), (x1 + lw, y1),
                    color, -1)
                cv2.putText(
                    annotated, label_text, (x1, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            live_fps = 1.0 / frame_times[-1]
            cv2.putText(
                annotated,
                f"Frame {frame_idx + 1}/{total}  |  {live_fps:.1f} FPS",
                (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)

            writer.write(annotated)
            frame_idx += 1

            if frame_idx % 50 == 0:
                elapsed = sum(frame_times)
                avg_fps = frame_idx / elapsed
                eta     = (elapsed / frame_idx) * (total - frame_idx)
                print(f"  Frame {frame_idx}/{total}  |  avg {avg_fps:.1f} FPS  |  ETA {eta:.0f}s")

    cap.release()
    writer.release()

    if not frame_times:
        print("No frames were processed.")
        return {}

    times_ms  = [t * 1000 for t in frame_times]
    avg_ms    = sum(times_ms) / len(times_ms)
    median_ms = float(np.median(times_ms))
    p95_ms    = float(np.percentile(times_ms, 95))
    avg_fps   = 1000 / avg_ms
    min_fps   = 1000 / max(times_ms)
    max_fps   = 1000 / min(times_ms)

    print("\n─── SSD Inference Results ──────────────────────────────")
    print(f"Output saved:      {video_out}")
    print(f"Frames processed:  {frame_idx}")
    print(f"Total detections:  {total_dets}")
    print(f"Avg inference:     {avg_ms:.2f} ms/frame")
    print(f"Median inference:  {median_ms:.2f} ms/frame")
    print(f"P95 inference:     {p95_ms:.2f} ms/frame   (95th percentile latency)")
    print(f"Avg FPS:           {avg_fps:.1f}")
    print(f"Min FPS:           {min_fps:.1f}  (slowest frame)")
    print(f"Max FPS:           {max_fps:.1f}  (fastest frame)")
    print("────────────────────────────────────────────────────────")

    return {
        'frames':     frame_idx,
        'detections': total_dets,
        'avg_ms':     round(avg_ms,    2),
        'median_ms':  round(median_ms, 2),
        'p95_ms':     round(p95_ms,    2),
        'avg_fps':    round(avg_fps,   1),
        'min_fps':    round(min_fps,   1),
        'max_fps':    round(max_fps,   1),
    }


if __name__ == '__main__':
    metrics = run_inference_on_video(WEIGHTS, VIDEO_IN, VIDEO_OUT)
