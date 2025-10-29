import os
import cv2
import albumentations as A
import numpy as np

# --- 1. CONFIGURATION ---

# Set the paths to your directories
IMAGE_DIR = os.path.join('train', 'images')
LABEL_DIR = os.path.join('train', 'labels')

# Set the paths for the new augmented data
OUTPUT_IMAGE_DIR = 'augmented_data/images'
OUTPUT_LABEL_DIR = 'augmented_data/labels'

# How many new augmented versions to create *per original image*
NUM_AUGMENTATIONS_PER_IMAGE = 5

# --- 2. HELPER FUNCTIONS ---

def read_yolo_labels(label_path):
    """Reads a YOLO .txt label file and returns bboxes and class_labels."""
    bboxes = []
    class_labels = []
    if not os.path.exists(label_path):
        return bboxes, class_labels

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                bboxes.append([x_center, y_center, width, height])
                class_labels.append(class_id)
    return bboxes, class_labels

def save_yolo_labels(label_path, bboxes, class_labels):
    """Saves bboxes and class_labels to a YOLO .txt label file."""
    with open(label_path, 'w') as f:
        for bbox, class_id in zip(bboxes, class_labels):
            x_center, y_center, width, height = bbox
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

# --- 3. AUGMENTATION PIPELINE ---

# Define your augmentation pipeline
# BboxParams tells albumentations how to handle bounding boxes
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=30, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.7),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    A.ColorJitter(p=0.5),
    # You can add many more! e.g., A.Cutout, A.CLAHE, A.RGBShift
], 
bbox_params=A.BboxParams(
    format='yolo',                # This is crucial for your .txt files
    label_fields=['class_labels'] # Tells it to also transform the labels
))

# --- 4. MAIN AUGMENTATION LOOP ---

def main():
    # Create output directories if they don't exist
    os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

    image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Found {len(image_files)} images. Starting augmentation...")

    for image_name in image_files:
        # Construct full paths
        image_path = os.path.join(IMAGE_DIR, image_name)
        
        # Get corresponding label path (e.g., image1.jpg -> image1.txt)
        base_name = os.path.splitext(image_name)[0]
        label_name = f"{base_name}.txt"
        label_path = os.path.join(LABEL_DIR, label_name)

        # Read image (use OpenCV)
        # We read in BGR and convert to RGB for albumentations
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not read image {image_path}, skipping.")
            continue
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Read labels
        bboxes, class_labels = read_yolo_labels(label_path)

        # --- Loop to create multiple augmentations ---
        for i in range(NUM_AUGMENTATIONS_PER_IMAGE):
            try:
                # Apply the transformation
                augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
                
                aug_image = augmented['image']
                aug_bboxes = augmented['bboxes']
                aug_class_labels = augmented['class_labels']

                # --- Create new filenames ---
                new_image_name = f"{base_name}_aug_{i}.jpg"
                new_label_name = f"{base_name}_aug_{i}.txt"
                
                new_image_path = os.path.join(OUTPUT_IMAGE_DIR, new_image_name)
                new_label_path = os.path.join(OUTPUT_LABEL_DIR, new_label_name)

                # --- Save the new data ---
                # Convert back to BGR for saving with OpenCV
                aug_image_bgr = cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(new_image_path, aug_image_bgr)
                
                # Save new labels
                save_yolo_labels(new_label_path, aug_bboxes, aug_class_labels)

            except Exception as e:
                print(f"Error augmenting {image_name} (iteration {i}): {e}")
                # This can sometimes happen if all bboxes are cropped out
                # You can choose to save the image with no labels if you want

        print(f"Generated {NUM_AUGMENTATIONS_PER_IMAGE} augmentations for {image_name}")

    print("---")
    print("Augmentation complete!")
    print(f"New images saved to: {OUTPUT_IMAGE_DIR}")
    print(f"New labels saved to: {OUTPUT_LABEL_DIR}")
    print(f"Total new images generated: {len(image_files) * NUM_AUGMENTATIONS_PER_IMAGE}")


if __name__ == "__main__":
    main()