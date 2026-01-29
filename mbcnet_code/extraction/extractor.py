"""Feature extraction from trained models"""
import os
import json
import pickle
import torch
from PIL import Image
from tqdm import tqdm
from data.transforms import get_transforms
from models import build_model


class IrisFeatureExtractor:
    """
    Extract features from iris images using trained model

    Args:
        model_path: Path to trained model weights
        config_path: Path to model configuration
        device: Device to use
    """
    def __init__(self, model_path=None, config_path=None, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # Set default paths
        if model_path is None:
            model_path = "outputs_simple/iris_model.pth"
        if config_path is None:
            config_path = "outputs_simple/model_config.json"

        # Load configuration
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")

        with open(config_path, 'r') as f:
            self.config = json.load(f)
        print(f"✓ Loaded config: {config_path}")

        # Load model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Build model from config
        config_obj = type('Config', (), self.config)()
        self.model = build_model(config_obj)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        print(f"✓ Loaded model: {model_path}")
        print(f"✓ Feature dimension: {self.config['feature_dim']}D")

        # Setup transforms
        self.transform = get_transforms(
            self.config['img_height'],
            self.config['img_width'],
            augmentation=False
        )

    def extract_from_image(self, image_path):
        """Extract feature from single image"""
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feature = self.model(img_tensor, return_embedding=False).cpu().numpy()

        return feature.squeeze()

    def extract_from_batch(self, image_paths):
        """Extract features from multiple images"""
        features = []
        for img_path in tqdm(image_paths, desc="Extracting", unit="image"):
            feature = self.extract_from_image(img_path)
            features.append(feature)
        return features

    def extract_from_folder(self, folder_path, save_output=None):
        """
        Extract features from all images in folder

        Args:
            folder_path: Path to folder containing images
            save_output: Optional path to save features (e.g., 'features.pkl')

        Returns:
            dict: {filename: feature_vector}
        """
        features = {}
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']

        image_files = [
            f for f in os.listdir(folder_path)
            if any(f.lower().endswith(ext) for ext in image_extensions)
        ]

        print(f"\n📁 Processing: {folder_path}")
        print(f"Found {len(image_files)} images")

        for filename in tqdm(image_files, desc="Processing", unit="image"):
            img_path = os.path.join(folder_path, filename)
            try:
                feature = self.extract_from_image(img_path)
                features[filename] = feature
            except Exception as e:
                print(f"⚠ Error with {filename}: {e}")

        print(f"✓ Extracted {len(features)} features")

        if save_output:
            self.save_features(features, save_output)

        return features

    def extract_from_dataset(self, dataset_path, save_output=None):
        """
        Extract features from entire dataset with class folders

        Args:
            dataset_path: Path to dataset (ImageFolder structure)
            save_output: Optional path to save features

        Returns:
            dict: {class_name: {filename: feature_vector}}
        """
        dataset_features = {}

        print(f"\n📂 Processing dataset: {dataset_path}")

        class_folders = [
            d for d in os.listdir(dataset_path)
            if os.path.isdir(os.path.join(dataset_path, d))
        ]

        print(f"Found {len(class_folders)} classes")

        for class_name in tqdm(class_folders, desc="Classes", unit="class"):
            class_path = os.path.join(dataset_path, class_name)
            class_features = self.extract_from_folder(class_path)
            dataset_features[class_name] = class_features

        total = sum(len(v) for v in dataset_features.values())
        print(f"✓ Total features: {total}")

        if save_output:
            self.save_features(dataset_features, save_output)

        return dataset_features

    def compare(self, feature1, feature2):
        """
        Compare two features using cosine similarity

        Args:
            feature1: First feature vector
            feature2: Second feature vector

        Returns:
            float: Similarity score (higher = more similar)
        """
        import numpy as np
        return float(np.dot(feature1, feature2))

    def verify(self, image_path1, image_path2, threshold=0.5):
        """
        Verify if two iris images are from same person

        Args:
            image_path1: Path to first image
            image_path2: Path to second image
            threshold: Similarity threshold

        Returns:
            tuple: (match: bool, similarity: float)
        """
        feat1 = self.extract_from_image(image_path1)
        feat2 = self.extract_from_image(image_path2)
        similarity = self.compare(feat1, feat2)
        match = similarity >= threshold
        return match, similarity

    def save_features(self, features, output_path):
        """Save features to file"""
        with open(output_path, 'wb') as f:
            pickle.dump(features, f)
        print(f"✓ Saved: {output_path}")

    def load_features(self, input_path):
        """Load features from file"""
        with open(input_path, 'rb') as f:
            features = pickle.load(f)
        print(f"✓ Loaded: {input_path}")
        return features