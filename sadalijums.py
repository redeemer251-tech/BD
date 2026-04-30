import os, shutil, random

src_img_dir = 'C:/Users/Admin/Desktop/BAKAULARA DARBS/dataset/combined/images/val/'
src_lab_dir = 'C:/Users/Admin/Desktop/BAKAULARA DARBS/dataset/combined/labels/val/'

base_dest = 'C:/Users/Admin/Desktop/BAKAULARA DARBS/dataset_final/'

images = [f for f in os.listdir(src_img_dir) if f.lower().endswith('.jpg')]
n = len(images)

if n == 0:
    print(f"Error: No images found in {src_img_dir}. Check your path!")
else:
    print(f"Found {n} images. Starting split...")
    random.shuffle(images)

    splits = {
        'train': images[:int(0.7*n)],
        'val':   images[int(0.7*n):int(0.85*n)],
        'test':  images[int(0.85*n):]
    }

    for split, files in splits.items():
        img_dest = os.path.join(base_dest, 'images', split)
        lab_dest = os.path.join(base_dest, 'labels', split)
        
        os.makedirs(img_dest, exist_ok=True)
        os.makedirs(lab_dest, exist_ok=True)
        
        for f in files:
            shutil.copy(os.path.join(src_img_dir, f), os.path.join(img_dest, f))
            
            label = f.replace('.jpg', '.txt').replace('.JPG', '.txt')
            if os.path.exists(os.path.join(src_lab_dir, label)):
                shutil.copy(os.path.join(src_lab_dir, label), os.path.join(lab_dest, label))

    print(f"Done! Check your folders at: {base_dest}")