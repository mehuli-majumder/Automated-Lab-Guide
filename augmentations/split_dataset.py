import os
import shutil
import random

print("Starting dataset split...")

# --- 1. CONFIGURATION ---
# Define the ratios for train, validation, and test sets
TRAIN_RATIO = 0.8
VALID_RATIO = 0.1
TEST_RATIO = 0.1  # These must sum to 1.0

# Define paths
INPUT_DIR = 'master_data'
OUTPUT_DIR = 'final_dataset'

IMAGE_DIR = os.path.join(INPUT_DIR, 'images')
LABEL_DIR = os.path.join(INPUT_DIR, 'labels')
YAML_FILE = os.path.join(INPUT_DIR, 'data.yaml')

# --- 2. CREATE OUTPUT DIRECTORIES ---
print(f"Creating output directory: {OUTPUT_DIR}")
for split in ['train', 'valid', 'test']:
    os.makedirs(os.path.join(OUTPUT_DIR, split, 'images'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, split, 'labels'), exist_ok=True)

# --- 3. GET & SHUFFLE ALL IMAGE FILES ---
all_images = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.png', '.jpeg'))]
random.shuffle(all_images)
total_count = len(all_images)
print(f"Found {total_count} total images to split.")

# --- 4. CALCULATE SPLIT COUNTS ---
train_count = int(total_count * TRAIN_RATIO)
valid_count = int(total_count * VALID_RATIO)
# test_count is the remainder

# --- 5. ASSIGN FILES TO SPLITS ---
train_files = all_images[:train_count]
valid_files = all_images[train_count : train_count + valid_count]
test_files = all_images[train_count + valid_count :]

print(f"Training images: {len(train_files)}")
print(f"Validation images: {len(valid_files)}")
print(f"Test images: {len(test_files)}")

# --- 6. HELPER FUNCTION TO COPY FILES ---
def copy_files(file_list, split_name):
    copied_count = 0
    for filename in file_list:
        basename = os.path.splitext(filename)[0]
        labelname = f"{basename}.txt"
        
        src_img = os.path.join(IMAGE_DIR, filename)
        src_label = os.path.join(LABEL_DIR, labelname)
        
        dest_img = os.path.join(OUTPUT_DIR, split_name, 'images', filename)
        dest_label = os.path.join(OUTPUT_DIR, split_name, 'labels', labelname)
        
        # Copy image
        shutil.copyfile(src_img, dest_img)
        
        # Copy label file *if it exists*
        if os.path.exists(src_label):
            shutil.copyfile(src_label, dest_label)
        else:
            # If no label file, create an empty one (common for background images)
            print(f"Info: No label file found for {filename}, creating empty label file.")
            open(dest_label, 'w').close()
        
        copied_count += 1
    print(f"Successfully copied {copied_count} files to {split_name} split.")

# --- 7. EXECUTE COPYING ---
copy_files(train_files, 'train')
copy_files(valid_files, 'valid')
copy_files(test_files, 'test')

# --- 8. COPY THE YAML FILE ---
dest_yaml = os.path.join(OUTPUT_DIR, 'data.yaml')
shutil.copyfile(YAML_FILE, dest_yaml)
print(f"Successfully copied data.yaml to {OUTPUT_DIR}")

print("---")
print("Dataset split complete!")
print(f"Your new dataset is ready in the '{OUTPUT_DIR}' folder.")