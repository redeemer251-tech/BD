import os
import xml.etree.ElementTree as ET

XML_DIR    = 'C:/Users/Admin/Desktop/BAKAULARA DARBS/dataset/uav/combinedxml'
IMAGES_DIR = 'C:/Users/Admin/Desktop/BAKAULARA DARBS/dataset/combined/images/val'
OUTPUT_DIR = 'C:/Users/Admin/Desktop/BAKAULARA DARBS/dataset/uav/combinedtxt'
CLASS_NAMES = ['uav']

os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_class_id(name, class_names):
    name = name.lower().strip()
    if name in class_names:
        return class_names.index(name)
    return None

def convert_box_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    """Convert absolute pixel coords to normalized YOLO format."""
    x_center = ((xmin + xmax) / 2) / img_w
    y_center = ((ymin + ymax) / 2) / img_h
    width    = (xmax - xmin) / img_w
    height   = (ymax - ymin) / img_h
    return x_center, y_center, width, height

def parse_pascal_voc(tree):
    """Parse standard Pascal VOC format."""
    root = tree.getroot()
    size = root.find('size')
    img_w = int(size.find('width').text)
    img_h = int(size.find('height').text)
    filename = root.find('filename').text

    boxes = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        class_id = get_class_id(name, CLASS_NAMES)
        if class_id is None:
            print(f"  Warning: unknown class '{name}' — skipping")
            continue
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        boxes.append((class_id, xmin, ymin, xmax, ymax))

    return filename, img_w, img_h, boxes

def parse_dut_custom(tree):
    """Parse DUT Anti-UAV custom XML format."""
    root = tree.getroot()

    image_nodes = root.findall('.//image')
    results = []

    for img_node in image_nodes:
        filename = img_node.get('name')
        img_w    = int(img_node.get('width'))
        img_h    = int(img_node.get('height'))

        boxes = []
        for box in img_node.findall('box'):
            name = box.get('label')
            class_id = get_class_id(name, CLASS_NAMES)
            if class_id is None:
                print(f"  Warning: unknown class '{name}' — skipping")
                continue
            xmin = float(box.get('xtl'))
            ymin = float(box.get('ytl'))
            xmax = float(box.get('xbr'))
            ymax = float(box.get('ybr'))
            boxes.append((class_id, xmin, ymin, xmax, ymax))

        results.append((filename, img_w, img_h, boxes))

    return results

def write_yolo_label(out_path, boxes, img_w, img_h):
    lines = []
    for (class_id, xmin, ymin, xmax, ymax) in boxes:
        xc, yc, w, h = convert_box_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h)
        xc = max(0.0, min(1.0, xc))
        yc = max(0.0, min(1.0, yc))
        w  = max(0.0, min(1.0, w))
        h  = max(0.0, min(1.0, h))
        lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines))

converted = 0
skipped   = 0

for xml_file in sorted(os.listdir(XML_DIR)):
    if not xml_file.endswith('.xml'):
        continue

    xml_path = os.path.join(XML_DIR, xml_file)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    file_stem = os.path.splitext(xml_file)[0]
    out_path = os.path.join(OUTPUT_DIR, file_stem + '.txt')

    is_pascal_voc = root.find('size') is not None

    if is_pascal_voc:
        _, img_w, img_h, boxes = parse_pascal_voc(tree)
        write_yolo_label(out_path, boxes, img_w, img_h)
        converted += 1

    else:
        records = parse_dut_custom(tree)
        for i, (internal_filename, img_w, img_h, boxes) in enumerate(records):
            if len(records) > 1:
                current_out_path = os.path.join(OUTPUT_DIR, f"{file_stem}_{i}.txt")
            else:
                current_out_path = out_path
                
            write_yolo_label(current_out_path, boxes, img_w, img_h)
            converted += 1

print(f"\nDone. Converted: {converted} files, Skipped: {skipped}")

import cv2
import numpy as np

def verify_conversion(images_dir, labels_dir, num_samples=5):
    image_files = [f for f in os.listdir(images_dir)
                   if f.lower().endswith(('.jpg', '.png', '.jpeg'))][:num_samples]

    for img_file in image_files:
        stem     = os.path.splitext(img_file)[0]
        img_path = os.path.join(images_dir, img_file)
        lbl_path = os.path.join(labels_dir, stem + '.txt')

        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read {img_file}")
            continue

        h, w = img.shape[:2]

        if not os.path.exists(lbl_path):
            print(f"No label found for {img_file}")
            continue

        with open(lbl_path) as f:
            lines = f.readlines()

        print(f"{img_file}: {len(lines)} object(s)")

        for line in lines:
            parts = line.strip().split()
            cls_id = int(parts[0])
            xc, yc, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

            x1 = int((xc - bw/2) * w)
            y1 = int((yc - bh/2) * h)
            x2 = int((xc + bw/2) * w)
            y2 = int((yc + bh/2) * h)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, CLASS_NAMES[cls_id], (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow('Verification', img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

verify_conversion(IMAGES_DIR, OUTPUT_DIR, num_samples=5)