import os
import shutil
from pathlib import Path
import random
from tqdm import tqdm

def create_directory_structure(base_path):
    """Create the necessary directory structure for the split dataset."""
    # Create main directories
    splits = ['train', 'val', 'test']
    classes = ['benign', 'malignant']
    
    # Create all directories
    for split in splits:
        for class_name in classes:
            os.makedirs(os.path.join(base_path, split, class_name), exist_ok=True)

def combine_and_split_data(source_path, target_path, train_ratio=0.75, val_ratio=0.15, test_ratio=0.10):
    """
    Combine train and test folders and split into train, validation, and test sets.
    
    Args:
        source_path: Path to the original dataset
        target_path: Path where the new split dataset will be created
        train_ratio: Proportion of data for training (default: 0.75)
        val_ratio: Proportion of data for validation (default: 0.15)
        test_ratio: Proportion of data for testing (default: 0.10)
    """
    # Create directory structure
    create_directory_structure(target_path)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Process each class
    for class_name in ['benign', 'malignant']:
        print(f"\nProcessing {class_name} class...")
        
        # Get all images from both train and test folders
        train_images = [f for f in os.listdir(os.path.join(source_path, 'train', class_name)) 
                       if f.endswith(('.jpg', '.jpeg', '.png'))]
        test_images = [f for f in os.listdir(os.path.join(source_path, 'test', class_name)) 
                      if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Combine all images
        all_images = train_images + test_images
        random.shuffle(all_images)
        
        # Calculate split sizes
        total_images = len(all_images)
        train_size = int(total_images * train_ratio)
        val_size = int(total_images * val_ratio)
        
        # Split the data
        train_images = all_images[:train_size]
        val_images = all_images[train_size:train_size + val_size]
        test_images = all_images[train_size + val_size:]
        
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
        
        for split_name, images in splits.items():
            print(f"\nCopying {split_name} images for {class_name}...")
            for img in tqdm(images):
                # Determine source path (either train or test)
                if img in train_images:
                    source_dir = 'train'
                else:
                    source_dir = 'test'
                
                src_path = os.path.join(source_path, source_dir, class_name, img)
                dst_path = os.path.join(target_path, split_name, class_name, img)
                shutil.copy2(src_path, dst_path)

if __name__ == "__main__":
    # Define paths
    source_path = "melanoma_cancer_dataset"
    target_path = "melanoma_cancer_dataset_split"
    
    # Perform the split
    combine_and_split_data(source_path, target_path)
    
    print("\nDataset split completed successfully!")
    print(f"New dataset structure created at: {target_path}") 