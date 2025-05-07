import os
import shutil
from pathlib import Path
from tqdm import tqdm

def combine_data(source_path, target_path):
    """
    Combine train and test folders into a single folder for each class.
    
    Args:
        source_path: Path to the original dataset containing train and test folders
        target_path: Path where the combined dataset will be created
    """
    # Create target directories
    classes = ['benign', 'malignant']
    for class_name in classes:
        os.makedirs(os.path.join(target_path, class_name), exist_ok=True)
    
    # Process each class
    for class_name in classes:
        print(f"\nProcessing {class_name} class...")
        
        # Get paths for train and test folders
        train_path = os.path.join(source_path, 'train', class_name)
        test_path = os.path.join(source_path, 'test', class_name)
        
        # Verify directories exist
        if not os.path.exists(train_path):
            print(f"Warning: Train directory not found: {train_path}")
            continue
        if not os.path.exists(test_path):
            print(f"Warning: Test directory not found: {test_path}")
            continue
        
        # Get all images from both folders
        train_images = [f for f in os.listdir(train_path) 
                       if f.endswith(('.jpg', '.jpeg', '.png'))]
        test_images = [f for f in os.listdir(test_path) 
                      if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Print counts
        print(f"Train images: {len(train_images)}")
        print(f"Test images: {len(test_images)}")
        print(f"Total images: {len(train_images) + len(test_images)}")
        
        # Copy train images
        print(f"\nCopying train images for {class_name}...")
        for img in tqdm(train_images):
            try:
                src_path = os.path.join(train_path, img)
                dst_path = os.path.join(target_path, class_name, img)
                shutil.copy2(src_path, dst_path)
            except Exception as e:
                print(f"Error copying {img}: {str(e)}")
        
        # Copy test images
        print(f"\nCopying test images for {class_name}...")
        for img in tqdm(test_images):
            try:
                src_path = os.path.join(test_path, img)
                dst_path = os.path.join(target_path, class_name, img)
                shutil.copy2(src_path, dst_path)
            except Exception as e:
                print(f"Error copying {img}: {str(e)}")

def verify_combined_data(target_path):
    """Verify the combined dataset by counting images in each class."""
    print("\nVerifying combined dataset:")
    print("-" * 50)
    
    for class_name in ['benign', 'malignant']:
        path = os.path.join(target_path, class_name)
        if os.path.exists(path):
            images = [f for f in os.listdir(path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            print(f"{class_name.capitalize()}: {len(images)} images")
        else:
            print(f"{class_name.capitalize()}: Directory not found!")
    
    print("-" * 50)

if __name__ == "__main__":
    # Define paths
    source_path = "melanoma_cancer_dataset"
    target_path = "melanoma_cancer_dataset_combined"
    
    # Create target directory if it doesn't exist
    os.makedirs(target_path, exist_ok=True)
    
    # Combine the data
    combine_data(source_path, target_path)
    
    # Verify the combined dataset
    verify_combined_data(target_path)
    
    print("\nData combination completed successfully!")
    print(f"Combined dataset created at: {target_path}") 