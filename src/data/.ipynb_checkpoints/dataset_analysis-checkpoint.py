import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

def analyze_dataset(base_path):
    # Initialize counters
    total_images = 0
    image_sizes = []
    aspect_ratios = []
    
    # Analyze both train and test sets
    for dataset_type in ['train', 'test']:
        for class_type in ['benign', 'malignant']:
            path = os.path.join(base_path, dataset_type, class_type)
            if not os.path.exists(path):
                continue
                
            # Count images
            images = [f for f in os.listdir(path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            total_images += len(images)
            
            # Analyze first 100 images from each class
            for img_name in images[:100]:
                img_path = os.path.join(path, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    height, width = img.shape[:2]
                    image_sizes.append((width, height))
                    aspect_ratios.append(width/height)
    
    return total_images, image_sizes, aspect_ratios

def plot_image_statistics(image_sizes, aspect_ratios):
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot image sizes
    widths, heights = zip(*image_sizes)
    ax1.scatter(widths, heights, alpha=0.5)
    ax1.set_xlabel('Width (pixels)')
    ax1.set_ylabel('Height (pixels)')
    ax1.set_title('Image Dimensions Distribution')
    
    # Plot aspect ratios
    sns.histplot(aspect_ratios, bins=30, ax=ax2)
    ax2.set_xlabel('Aspect Ratio (width/height)')
    ax2.set_ylabel('Count')
    ax2.set_title('Aspect Ratio Distribution')
    
    plt.tight_layout()
    plt.savefig('results/image_statistics.png')
    plt.close()

def display_sample_images(base_path, num_samples=5):
    fig, axes = plt.subplots(2, num_samples, figsize=(20, 8))
    
    for i, class_type in enumerate(['benign', 'malignant']):
        path = os.path.join(base_path, 'train', class_type)
        images = [f for f in os.listdir(path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        for j in range(num_samples):
            img_path = os.path.join(path, images[j])
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            axes[i, j].imshow(img)
            axes[i, j].axis('off')
            if j == 0:
                axes[i, j].set_ylabel(class_type.capitalize())
    
    plt.tight_layout()
    plt.savefig('results/sample_images.png')
    plt.close()

if __name__ == "__main__":
    base_path = "melanoma_cancer_dataset"
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Analyze dataset
    total_images, image_sizes, aspect_ratios = analyze_dataset(base_path)
    print(f"Total number of images: {total_images}")
    
    # Plot statistics
    plot_image_statistics(image_sizes, aspect_ratios)
    
    # Display sample images
    display_sample_images(base_path)
    
    print("Analysis complete! Check the 'results' directory for visualizations.") 