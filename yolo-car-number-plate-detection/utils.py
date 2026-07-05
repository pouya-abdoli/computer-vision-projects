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
    # Get class names from YAML config
    class_names = data_config["names"]
    num_classes = len(class_names)
    
    # Default colors if not provided
    if colors is None:
        default_colors = ['red', 'blue', 'green', 'orange', 'purple', 
                         'cyan', 'magenta', 'yellow', 'brown', 'pink']
        colors = default_colors[:num_classes]
    
    # Get images directory path for the specified split
    split_path = data_config[split]
    train_path = Path(split_path) / 'images'
    
    # Get all image paths (support multiple extensions)
    all_images_path = []
    all_images_path.extend(glob.glob(f"{train_path}/*.png"))
    all_images_path.extend(glob.glob(f"{train_path}/*.jpg"))
    all_images_path.extend(glob.glob(f"{train_path}/*.jpeg"))
    
    # Check if images exist
    if not all_images_path:
        print(f"No images found in {train_path}")
        return
    
    # Sample random images
    sample_count = min(num_samples, len(all_images_path))
    sample_images = random.choices(all_images_path, k=sample_count)
    
    for img_path in sample_images:
        # Load and Convert image (BGR to RGB)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get image dimensions: height, width (ignore channels)
        h, w, _ = img.shape
        
        # Convert string path to Path object for easy manipulation
        image_path = Path(img_path)
        
        # Goes from: split/images/image.jpg → split/labels/image.txt
        label_path = image_path.parent.parent / "labels" / (image_path.stem + ".txt")
        
        # If label exists: draw bounding boxes
        if label_path.exists():
            plt.figure(figsize=(10, 10))
            plt.imshow(img)
            ax = plt.gca()
            
            # Parse YOLO format: class x_center y_center width height (normalized 0-1)
            with open(label_path) as f:
                for line in f:
                    cls, xc, yc, bw, bh = map(float, line.split())
                    class_id = int(cls)
                    
                    # Get color for this class
                    color = colors[class_id % len(colors)]
                    
                    # Scale normalized YOLO coordinates to absolute pixel values
                    x1 = (xc - bw/2) * w
                    x2 = (xc + bw/2) * w
                    y1 = (yc - bh/2) * h
                    y2 = (yc + bh/2) * h
                    
                    # Create rectangle
                    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1,
                                         fill=False,
                                         edgecolor=color,
                                         linewidth=2)
                    ax.add_patch(rect)
                    
                    # Draw class label
                    plt.text(x1, y1-10,
                             class_names[class_id],
                             color='white',
                             fontsize=10,
                             bbox=dict(facecolor=color, alpha=0.9))
            
            plt.axis('off')
            plt.show()
        else:
            print(f"No label found for: {img_path}")