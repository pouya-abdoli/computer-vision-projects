import glob
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import random

def visualize_annotations(data_config, num_samples=5, split='train', colors=None):
    """
    Visualize random sample images with their YOLO format bounding boxes.
    
    Args:
        data_config: Dictionary containing 'names', and split paths
        num_samples: Number of random images to display (default: 5)
        split: Dataset split to visualize - 'train', 'val', or 'test' (default: 'train')
        colors: List of colors for each class. If None, uses default colors
    """
    # Get class names
    class_names = data_config["names"]
    num_classes = len(class_names)
    
    # Default colors
    if colors is None:
        default_colors = ['red', 'blue', 'green', 'orange', 'purple', 
                         'cyan', 'magenta', 'yellow', 'brown', 'pink']
        colors = default_colors[:num_classes]
    
    # Get images path
    split_path = Path(data_config[split])
    images_path = split_path / 'images'
    
    # Get all images
    all_images = []
    all_images.extend(glob.glob(f"{images_path}/*.png"))
    all_images.extend(glob.glob(f"{images_path}/*.jpg"))
    all_images.extend(glob.glob(f"{images_path}/*.jpeg"))
    
    if not all_images:
        print(f"No images found in {images_path}")
        return
    
    # Sample random images
    sample_count = min(num_samples, len(all_images))
    sample_images = random.choices(all_images, k=sample_count)
    
    # Process each image
    for img_path in sample_images:
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load: {img_path}")
            continue
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        
        # Get label path
        img_file = Path(img_path)
        label_path = img_file.parent.parent / "labels" / (img_file.stem + ".txt")
        
        if not label_path.exists():
            print(f"No label: {img_path}")
            continue
        
        # Create plot
        plt.figure(figsize=(10, 10))
        plt.imshow(img)
        ax = plt.gca()
        
        # Read labels
        with open(label_path) as f:
            for line in f:
                cls, xc, yc, bw, bh = map(float, line.split())
                class_id = int(cls)
                color = colors[class_id % len(colors)]
                
                # Convert to pixel coordinates
                x1 = (xc - bw/2) * w
                x2 = (xc + bw/2) * w
                y1 = (yc - bh/2) * h
                y2 = (yc + bh/2) * h
                
                # Draw box
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                     fill=False, edgecolor=color, linewidth=2)
                ax.add_patch(rect)
                
                # Draw label
                plt.text(x1, y1-10, class_names[class_id],
                         color='white', fontsize=10,
                         bbox=dict(facecolor=color, alpha=0.9))
        
        # Show image name at bottom
        plt.axis('off')
        
        # Instead of figtext at bottom, put text on the image
        plt.text(10, 20, f"Image: {img_file.name}", 
                color='white', fontsize=12,
                bbox=dict(facecolor='black', alpha=0.7))
        plt.show()