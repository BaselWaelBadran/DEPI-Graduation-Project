import os
import shutil
from pathlib import Path
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import cv2

def create_split_directories(base_path):
    """Create the necessary directory structure for the split dataset."""
    # Create main directories
    splits = ['train', 'val', 'test']
    classes = ['benign', 'malignant']
    
    # Create all directories
    for split in splits:
        for class_name in classes:
            os.makedirs(os.path.join(base_path, split, class_name), exist_ok=True)

def split_combined_data(source_path, target_path, train_ratio=0.75, val_ratio=0.15, test_ratio=0.10):
    """
    Split the combined dataset into train, validation, and test sets.
    
    Args:
        source_path: Path to the combined dataset
        target_path: Path where the split dataset will be created
        train_ratio: Proportion of data for training (default: 0.75)
        val_ratio: Proportion of data for validation (default: 0.15)
        test_ratio: Proportion of data for testing (default: 0.10)
    """
    # Create directory structure
    create_split_directories(target_path)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Store split information for visualization
    split_info = {}
    
    # Process each class
    for class_name in ['benign', 'malignant']:
        print(f"\nProcessing {class_name} class...")
        
        # Get path for the class folder
        class_path = os.path.join(source_path, class_name)
        
        # Verify directory exists
        if not os.path.exists(class_path):
            print(f"Warning: Class directory not found: {class_path}")
            continue
        
        # Get all images
        images = [f for f in os.listdir(class_path) 
                 if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Shuffle images
        random.shuffle(images)
        
        # Calculate split sizes
        total_images = len(images)
        train_size = int(total_images * train_ratio)
        val_size = int(total_images * val_ratio)
        
        # Split the data
        train_images = images[:train_size]
        val_images = images[train_size:train_size + val_size]
        test_images = images[train_size + val_size:]
        
        # Store split information
        split_info[class_name] = {
            'total': total_images,
            'train': len(train_images),
            'val': len(val_images),
            'test': len(test_images)
        }
        
        # Print split information
        print(f"Total {class_name} images: {total_images}")
        print(f"Train: {len(train_images)} ({len(train_images)/total_images*100:.1f}%)")
        print(f"Validation: {len(val_images)} ({len(val_images)/total_images*100:.1f}%)")
        print(f"Test: {len(test_images)} ({len(test_images)/total_images*100:.1f}%)")
        
        # Copy files to their respective directories
        splits = {
            'train': train_images,
            'val': val_images,
            'test': test_images
        }
        
        for split_name, split_images in splits.items():
            print(f"\nCopying {split_name} images for {class_name}...")
            for img in tqdm(split_images):
                try:
                    src_path = os.path.join(class_path, img)
                    dst_path = os.path.join(target_path, split_name, class_name, img)
                    shutil.copy2(src_path, dst_path)
                except Exception as e:
                    print(f"Error copying {img}: {str(e)}")
    
    return split_info

def visualize_split(split_info, save_path='results/dataset_split_visualization.png'):
    """Visualize the dataset split."""
    # Prepare data for plotting
    classes = list(split_info.keys())
    splits = ['train', 'val', 'test']
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Bar chart of split sizes
    x = np.arange(len(classes))
    width = 0.25
    
    for i, split in enumerate(splits):
        values = [split_info[cls][split] for cls in classes]
        ax1.bar(x + i*width, values, width, label=split.capitalize())
    
    ax1.set_xlabel('Class')
    ax1.set_ylabel('Number of Images')
    ax1.set_title('Dataset Split by Class')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels([cls.capitalize() for cls in classes])
    ax1.legend()
    
    # Plot 2: Pie chart of overall split
    total_train = sum(split_info[cls]['train'] for cls in classes)
    total_val = sum(split_info[cls]['val'] for cls in classes)
    total_test = sum(split_info[cls]['test'] for cls in classes)
    
    ax2.pie([total_train, total_val, total_test], 
            labels=['Train', 'Validation', 'Test'],
            autopct='%1.1f%%',
            startangle=90)
    ax2.axis('equal')
    ax2.set_title('Overall Dataset Split')
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    
    print(f"Visualization saved to {save_path}")

def verify_split(target_path):
    """Verify the split by counting images in each directory."""
    splits = ['train', 'val', 'test']
    classes = ['benign', 'malignant']
    
    print("Verifying split:")
    print("-" * 50)
    
    for split in splits:
        print(f"\n{split.upper()} SET:")
        for class_name in classes:
            path = os.path.join(target_path, split, class_name)
            if os.path.exists(path):
                images = [f for f in os.listdir(path) if f.endswith(('.jpg', '.jpeg', '.png'))]
                print(f"  {class_name.capitalize()}: {len(images)} images")
            else:
                print(f"  {class_name.capitalize()}: Directory not found!")
    
    print("\n" + "-" * 50)

def display_sample_images(base_path, num_samples=5, random_seed=None):
    """
    Display random sample images from each class with titles and class labels.
    
    Args:
        base_path: Path to the dataset
        num_samples: Number of samples to display per class
        random_seed: Random seed for reproducibility (optional)
    """
    # Set random seed if provided
    if random_seed is not None:
        random.seed(random_seed)
    
    fig, axes = plt.subplots(2, num_samples, figsize=(20, 8))
    
    for i, class_type in enumerate(['benign', 'malignant']):
        path = os.path.join(base_path, 'train', class_type)
        images = [f for f in os.listdir(path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Randomly select samples
        selected_images = random.sample(images, min(num_samples, len(images)))
        
        for j, img_name in enumerate(selected_images):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            axes[i, j].imshow(img)
            axes[i, j].axis('off')
            
            # Add title with class and image name
            title = f"{class_type.capitalize()}\n{img_name}"
            axes[i, j].set_title(title, fontsize=10, pad=5)
            
            if j == 0:
                axes[i, j].set_ylabel(class_type.capitalize(), fontsize=12, fontweight='bold')
    
    plt.suptitle('Random Sample Images from Each Class', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Define paths
    source_path = "melanoma_cancer_dataset_combined"
    target_path = "melanoma_cancer_dataset_split"
    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Perform the split
    split_info = split_combined_data(source_path, target_path)
    
    # Visualize the split
    visualize_split(split_info)
    
    # Verify the split
    verify_split(target_path)
    
    print("\nDataset split completed successfully!")
    print(f"New dataset structure created at: {target_path}")

    # Display sample images
    display_sample_images(target_path) 