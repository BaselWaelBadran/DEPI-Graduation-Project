import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from pathlib import Path

def get_transforms(split='train'):
    """
    Get image transforms for training and validation/testing.
    
    Args:
        split (str): One of 'train', 'val', or 'test'
    
    Returns:
        transforms.Compose: Composed transforms
    """
    if split == 'train':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
    else:  # val or test
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])

def get_data_loaders(data_dir, batch_size=32, num_workers=4):
    """
    Create DataLoaders for train, validation, and test sets using ImageFolder.
    
    Args:
        data_dir (str): Root directory of the dataset
        batch_size (int): Batch size for the DataLoader
        num_workers (int): Number of workers for data loading
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Create datasets using ImageFolder
    train_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'train'),
        transform=get_transforms('train')
    )
    
    val_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'val'),
        transform=get_transforms('val')
    )
    
    test_dataset = datasets.ImageFolder(
        root=os.path.join(data_dir, 'test'),
        transform=get_transforms('test')
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True  # Speeds up data transfer to GPU
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Test the dataset and data loaders
    data_dir = "melanoma_cancer_dataset_split"
    
    # Create data loaders
    train_loader, val_loader, test_loader = get_data_loaders(
        data_dir=data_dir,
        batch_size=32,
        num_workers=4
    )
    
    # Print dataset sizes
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Print class distribution
    train_labels = [label for _, label in train_loader.dataset]
    val_labels = [label for _, label in val_loader.dataset]
    test_labels = [label for _, label in test_loader.dataset]
    
    print("\nClass distribution:")
    print(f"Train - Benign: {train_labels.count(0)}, Malignant: {train_labels.count(1)}")
    print(f"Val - Benign: {val_labels.count(0)}, Malignant: {val_labels.count(1)}")
    print(f"Test - Benign: {test_labels.count(0)}, Malignant: {test_labels.count(1)}")
    
    # Test GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Test loading a batch
    for images, labels in train_loader:
        print(f"\nBatch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        break 