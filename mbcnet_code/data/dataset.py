"""
Dataset loading and splitting utilities
"""
from torchvision import datasets
from torch.utils.data import DataLoader, random_split
from .transforms import get_transforms


def load_dataset(data_dir, img_height, img_width, train_split=0.8, 
                batch_size=64, augmentation=False, aug_config=None, 
                num_workers=4):
    """
    Load and split dataset

    Args:
        data_dir: Path to dataset
        img_height: Image height
        img_width: Image width
        train_split: Fraction of data for training
        batch_size: Batch size
        augmentation: Use data augmentation
        aug_config: Augmentation configuration
        num_workers: Number of data loading workers

    Returns:
        train_loader, test_loader, num_classes, train_size, test_size
    """
    # Get transforms
    train_transform = get_transforms(img_height, img_width, augmentation, aug_config)
    test_transform = get_transforms(img_height, img_width, False, None)

    # Load full dataset
    dataset = datasets.ImageFolder(data_dir, transform=train_transform)

    # Split dataset
    train_size = int(train_split * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Override test transform
    test_dataset.dataset.transform = test_transform

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
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

    num_classes = len(dataset.classes)

    return train_loader, test_loader, num_classes, train_size, test_size