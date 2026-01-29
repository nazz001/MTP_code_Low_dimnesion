"""
Image transformations and augmentations
"""
from torchvision import transforms


def get_transforms(img_height, img_width, augmentation=False, aug_config=None):
    """
    Get image transformation pipeline
    
    Args:
        img_height: Target image height
        img_width: Target image width
        augmentation: Whether to apply data augmentation
        aug_config: Augmentation configuration dict
    
    Returns:
        transforms.Compose object
    """
    transform_list = [
        transforms.Grayscale(1),
        transforms.Resize((img_height, img_width)),
    ]
    
    # Add augmentation if enabled
    if augmentation and aug_config:
        if aug_config.get("random_horizontal_flip", 0) > 0:
            transform_list.append(
                transforms.RandomHorizontalFlip(aug_config["random_horizontal_flip"])
            )
        
        if aug_config.get("random_rotation", 0) > 0:
            transform_list.append(
                transforms.RandomRotation(aug_config["random_rotation"])
            )
        
        if aug_config.get("random_brightness", 0) > 0 or aug_config.get("random_contrast", 0) > 0:
            transform_list.append(
                transforms.ColorJitter(
                    brightness=aug_config.get("random_brightness", 0),
                    contrast=aug_config.get("random_contrast", 0)
                )
            )
    
    # Convert to tensor and normalize
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    return transforms.Compose(transform_list)
