import os
import re

# Set this to the folder containing your 5200 .txt files
TARGET_DIR = r'C:\Users\Admin\Desktop\BAKAULARA DARBS\dataset\uav\val\txt'

def fix_class_id(directory):
    count = 0
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            
            with open(file_path, 'r') as f:
                content = f.read()

            # re.sub with count=1 replaces only the first occurrence found
            # This turns the first '0' (Class 0) into '2' (Class 2)
            new_content = re.sub(r'0', '2', content, count=1)

            with open(file_path, 'w') as f:
                f.write(new_content)
            
            count += 1

    print(f"Successfully updated {count} files.")

if __name__ == "__main__":
    fix_class_id(TARGET_DIR)