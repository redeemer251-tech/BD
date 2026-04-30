import cv2
import numpy as np
import os

def add_fog(img, intensity=0.5):
    fog = np.full_like(img, 255, dtype=np.uint8)
    return cv2.addWeighted(img, 1 - intensity, fog, intensity, 0)

def add_rain(img, num_drops=600, length=12, angle=-15):
    out = img.copy()
    h, w = img.shape[:2]
    for _ in range(num_drops):
        x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
        dx = int(length * np.cos(np.radians(angle)))
        dy = int(length * np.sin(np.radians(angle)))
        cv2.line(out, (x1, y1), (x1+dx, y1+dy), (200, 200, 200), 1)
    return cv2.addWeighted(img, 0.7, out, 0.3, 0)

def add_low_light(img, gamma=0.35):
    table = np.array([(i/255.0)**(1.0/gamma)*255 for i in range(256)]).astype(np.uint8)
    dark = cv2.LUT(img, table)
    noise = np.random.normal(0, 8, dark.shape).astype(np.int16)
    return np.clip(dark.astype(np.int16) + noise, 0, 255).astype(np.uint8)

src = 'dataset_final/images/test'
augmentations = {'fog': add_fog, 'rain': add_rain, 'lowlight': add_low_light}

for cond_name, fn in augmentations.items():
    out_dir = f'dataset_final/conditions/test_{cond_name}'
    os.makedirs(out_dir, exist_ok=True)
    for fname in os.listdir(src):
        img = cv2.imread(os.path.join(src, fname))
        aug = fn(img)
        cv2.imwrite(os.path.join(out_dir, fname), aug)