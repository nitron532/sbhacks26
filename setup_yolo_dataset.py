# setup_yolo_dataset.py
import json
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import shutil
import sys

# Add path to import constants
sys.path.append('KataCR')
from katacr.constants.label_list import unit_list, unit2idx

# Configuration
METADATA_DIR = Path("KataCR/logs/metadata")
IMAGES_DIR = Path("KataCR/logs/generation")
OUTPUT_BASE = Path("yolo_dataset")  # Change this to your desired location
IMG_WIDTH = 568
IMG_HEIGHT = 896

def xyxy_to_yolo(xyxy, img_width, img_height):
    """Convert xyxy to YOLO normalized format"""
    x1, y1, x2, y2 = xyxy
    center_x = (x1 + x2) / 2.0 / img_width
    center_y = (y1 + y2) / 2.0 / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    
    center_x = max(0, min(1, center_x))
    center_y = max(0, min(1, center_y))
    width = max(0, min(1, width))
    height = max(0, min(1, height))
    
    return center_x, center_y, width, height

def convert_json_to_yolo(json_path, labels_dir, img_width, img_height):
    """Convert JSON metadata to YOLO txt file"""
    with open(json_path, 'r') as f:
        metadata = json.load(f)
    
    label_filename = Path(json_path.stem).stem + '.txt'  # Remove .json
    label_path = labels_dir / label_filename
    
    with open(label_path, 'w') as f:
        for unit in metadata['units']:
            if unit['drop'] or unit['cls'] == -1:
                continue
            
            cls_id = unit['cls']
            xyxy = unit['xyxy']
            cx, cy, w, h = xyxy_to_yolo(xyxy, img_width, img_height)
            f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
    
    return label_filename

# Main conversion
print("Setting up YOLO dataset...")

# Get JSON files (exclude _box versions)
json_files = sorted([f for f in METADATA_DIR.glob('*.json') 
                     if not f.name.endswith('_box.json')])

print(f"Found {len(json_files)} JSON files")

# Create directory structure
for split in ['train', 'val', 'test']:
    (OUTPUT_BASE / split / 'images').mkdir(parents=True, exist_ok=True)
    (OUTPUT_BASE / split / 'labels').mkdir(parents=True, exist_ok=True)

# Split dataset
train_files, temp_files = train_test_split(json_files, test_size=0.2, random_state=42)
val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)

splits = {'train': train_files, 'val': val_files, 'test': test_files}

# Convert and copy files
for split_name, files in splits.items():
    labels_dir = OUTPUT_BASE / split_name / 'labels'
    images_dir_out = OUTPUT_BASE / split_name / 'images'
    
    for json_file in files:
        img_stem = json_file.stem
        img_file = IMAGES_DIR / f"{img_stem}.jpg"
        
        if not img_file.exists():
            print(f"Warning: {img_file} not found, skipping")
            continue
        
        convert_json_to_yolo(json_file, labels_dir, IMG_WIDTH, IMG_HEIGHT)
        shutil.copy2(img_file, images_dir_out / img_file.name)
    
    print(f"Processed {split_name}: {len(files)} files")

# Generate data.yaml
yaml_path = OUTPUT_BASE / 'data.yaml'
yaml_content = f"""# YOLO Dataset Configuration
path: {OUTPUT_BASE.absolute()}  # dataset root dir
train: train/images
val: val/images
test: test/images

# Number of classes
nc: {len(unit_list)}

# Class names
names:
"""

for idx, class_name in enumerate(unit_list):
    yaml_content += f"  {idx}: '{class_name}'\n"

with open(yaml_path, 'w') as f:
    f.write(yaml_content)

print(f"\nDataset setup complete!")
print(f"Dataset location: {OUTPUT_BASE.absolute()}")
print(f"Classes: {len(unit_list)}")
print(f"\nTrain: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")
print(f"\nRun training with:")
print(f"yolo task=detect mode=train model=yolo11n.pt data={yaml_path} epochs=35 imgsz=640 batch=16 device=0")