import os
import re

TARGET_DIR = r'C:\Users\Admin\Desktop\BAKAULARA DARBS\dataset\uav\combinedtxt'

def fix_class_id(directory):
    count = 0
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            
            with open(file_path, 'r') as f:
                content = f.read()

            new_content = re.sub(r'0', '2', content, count=1)

            with open(file_path, 'w') as f:
                f.write(new_content)
            
            count += 1

    print(f"Successfully updated {count} files.")

if __name__ == "__main__":
    fix_class_id(TARGET_DIR)