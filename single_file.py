import os
import random
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_curve, auc
from tqdm import tqdm
from PIL import Image


# ===================== CONFIGURATION =====================
# Dataset paths
casia_iris_thousand = "/home/nishkal/datasets/iris_db/CASIA_iris_thousand/worldcoin_outputs_images"
casia_v1 = "/home/nishkal/datasets/iris_db/CASIA_v1"
iitd_v1 = "/home/nishkal/datasets/iris_db/IITD_v1"

# Select dataset for training
DATA_DIR = casia_iris_thousand

# Output
OUTPUT_DIR = "outputs_simple"
MODEL_NAME = "iris_model.pth"
MODEL_CONFIG_NAME = "model_config.json"  # Save model architecture config
EMBEDDINGS_NAME = "embeddings.pkl"
RESULTS_NAME = "results.json"

# Image dimensions
IMG_HEIGHT = 60
IMG_WIDTH = 240

# Training parameters
SEED = 42
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-4
MARGIN = 0.3

# Model parameters
EMBED_DIM = 256
FEATURE_DIM = 128

# Data split
TRAIN_SPLIT = 0.8  # 80% train, 20% test

# Evaluation
MAX_PAIRS = 200000

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===================== SEED =====================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(SEED)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ===================== MODEL DEFINITION =====================
def conv_block(ic, oc, k, p=0, d=1):
    return nn.Sequential(
        nn.Conv2d(ic, oc, k, padding=p, dilation=d),
        nn.BatchNorm2d(oc),
        nn.ReLU(inplace=True)
    )


class ChannelAttn(nn.Module):
    def __init__(self, c, k=5):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, k, padding=k//2)
        self.act = nn.ReLU()
        self.sig = nn.Sigmoid()
    
    def forward(self, x):
        B, C, _, _ = x.shape
        f = x.mean((2, 3)) + x.view(B, C, -1).max(2)[0]
        w = self.sig(self.act(self.conv(f.unsqueeze(1))))
        return x * w.view(B, C, 1, 1)


class MBLNet(nn.Module):
    def __init__(self):
        super().__init__()
        chs = [16, 24, 32, 48, 64]
        self.branches = nn.ModuleList()
        
        for k, d in [(3, 1), (5, 1), (3, 2)]:
            layers = []
            for i, c in enumerate(chs):
                ic = 1 if i == 0 else chs[i-1]
                layers.append(conv_block(ic, c, k, p=(k//2)*d, d=d))
                layers.append(ChannelAttn(c))
                if i < len(chs) - 1:
                    layers.append(nn.AvgPool2d(2))
            layers.append(nn.AdaptiveAvgPool2d((1, 1)))
            self.branches.append(nn.Sequential(*layers))
        
        self.fc = nn.Sequential(
            nn.Linear(chs[-1]*3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, EMBED_DIM)
        )
        
        self.projection_head = nn.Sequential(
            nn.Linear(EMBED_DIM, EMBED_DIM),
            nn.BatchNorm1d(EMBED_DIM),
            nn.ReLU(inplace=True),
            nn.Linear(EMBED_DIM, FEATURE_DIM)
        )
    
    def forward(self, x, return_embedding=False):
        feats = []
        for b in self.branches:
            f = b(x).view(x.size(0), -1)
            feats.append(f)
        
        embedding = self.fc(torch.cat(feats, 1))
        embedding = embedding / (embedding.norm(dim=1, keepdim=True) + 1e-8)
        
        if return_embedding:
            return embedding
        
        features = self.projection_head(embedding)
        features = features / (features.norm(dim=1, keepdim=True) + 1e-8)
        
        return features


# ===================== LOSS FUNCTION =====================
def pairwise_dist(x):
    return 1 - torch.matmul(x, x.T)


def batch_hard_triplet(emb, lbl, margin):
    d = pairwise_dist(emb)
    lbl = lbl.unsqueeze(1)
    
    pos = d.clone()
    pos[lbl != lbl.T] = -1
    hardest_pos = pos.max(1)[0].clamp(min=0)
    
    neg = d.clone()
    neg[lbl == lbl.T] = 1e6
    hardest_neg = neg.min(1)[0]
    
    return torch.relu(margin + hardest_pos - hardest_neg).mean()


# ===================== EXTRACT EMBEDDINGS =====================
def extract_embeddings(model, loader):
    """Extract 128D features"""
    model.eval()
    E, L = [], []
    with torch.no_grad():
        for x, y in tqdm(loader, desc="Extracting features", leave=False, unit="batch"):
            z = model(x.to(DEVICE), return_embedding=False).cpu().numpy()
            E.append(z)
            L.append(y.numpy())
    return np.vstack(E), np.concatenate(L)


# ===================== EVALUATION =====================
def evaluate(test_emb, test_lbl, save_prefix="test"):
    sims, ytrue = [], []
    count = 0
    n = len(test_emb)
    
    total_pairs = min(MAX_PAIRS, n * (n - 1) // 2)
    
    with tqdm(total=total_pairs, desc="Evaluating", unit="pairs") as pbar:
        for i in range(n):
            for j in range(i+1, n):
                sims.append(np.dot(test_emb[i], test_emb[j]))
                ytrue.append(int(test_lbl[i] == test_lbl[j]))
                count += 1
                pbar.update(1)
                if count >= MAX_PAIRS:
                    break
            if count >= MAX_PAIRS:
                break
    
    sims = np.array(sims)
    ytrue = np.array(ytrue)
    
    fpr, tpr, thr = roc_curve(ytrue, sims)
    roc_auc = auc(fpr, tpr)
    fnr = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    
    print(f"\n📊 Results - AUC={roc_auc:.4f}, EER={eer:.4f}")
    
    # Save ROC plot
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}", linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.scatter(fpr[eer_idx], tpr[eer_idx], label=f"EER={eer:.4f}", color='red', s=100, zorder=5)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, f"roc_{save_prefix}.png"), dpi=150)
    plt.close()
    
    # Save FAR/FRR plot
    plt.figure(figsize=(10, 6))
    plt.plot(thr, fpr, label="FAR", linewidth=2)
    plt.plot(thr, fnr, label="FRR", linewidth=2)
    plt.axvline(thr[eer_idx], color='red', linestyle='--', 
                label=f'EER Threshold={thr[eer_idx]:.3f}', alpha=0.7)
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('Rate', fontsize=12)
    plt.title('FAR/FRR Curves', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, f"far_frr_{save_prefix}.png"), dpi=150)
    plt.close()
    
    return {
        "auc": float(roc_auc),
        "eer": float(eer),
        "eer_threshold": float(thr[eer_idx])
    }


# ===================== FEATURE EXTRACTOR CLASS =====================
class IrisFeatureExtractor:
    def __init__(self, model_path=None, config_path=None):
        """Initialize feature extractor with trained model"""
        self.device = DEVICE
        
        if model_path is None:
            model_path = os.path.join(OUTPUT_DIR, MODEL_NAME)
        if config_path is None:
            config_path = os.path.join(OUTPUT_DIR, MODEL_CONFIG_NAME)
        
        # Load model config
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            print(f"✓ Loaded config: {config_path}")
        else:
            # Default config
            self.config = {
                'img_height': IMG_HEIGHT,
                'img_width': IMG_WIDTH,
                'embed_dim': EMBED_DIM,
                'feature_dim': FEATURE_DIM
            }
            print(f"⚠ Config not found, using defaults")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.model = MBLNet().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"✓ Loaded model from: {model_path}")
        print(f"✓ Feature dimension: {self.config['feature_dim']}D")
        
        self.transform = transforms.Compose([
            transforms.Grayscale(1),
            transforms.Resize((self.config['img_height'], self.config['img_width'])),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
    
    def extract_from_image(self, image_path):
        """Extract 128D feature from a single image"""
        img = Image.open(image_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            feature = self.model(img_tensor, return_embedding=False).cpu().numpy()
        
        return feature.squeeze()
    
    def extract_from_batch(self, image_paths):
        """Extract features from multiple images"""
        features = []
        for img_path in tqdm(image_paths, desc="Extracting features", unit="image"):
            feature = self.extract_from_image(img_path)
            features.append(feature)
        return np.array(features)
    
    def extract_from_folder(self, folder_path, save_output=None):
        """
        Extract features from all images in a folder
        
        Args:
            folder_path: Path to folder containing images
            save_output: Optional path to save features (e.g., 'features.pkl')
        
        Returns:
            dict: {filename: feature_vector}
        """
        features = {}
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        image_files = [f for f in os.listdir(folder_path) 
                      if any(f.lower().endswith(ext) for ext in image_extensions)]
        
        print(f"\n📁 Processing folder: {folder_path}")
        print(f"Found {len(image_files)} images")
        
        for filename in tqdm(image_files, desc="Processing images", unit="image"):
            img_path = os.path.join(folder_path, filename)
            try:
                feature = self.extract_from_image(img_path)
                features[filename] = feature
            except Exception as e:
                print(f"⚠ Error processing {filename}: {e}")
        
        print(f"✓ Successfully extracted {len(features)} features")
        
        if save_output:
            self.save_features(features, save_output)
        
        return features
    
    def extract_from_dataset(self, dataset_path, save_output=None):
        """
        Extract features from entire dataset (ImageFolder structure)
        
        Args:
            dataset_path: Path to dataset with class folders
            save_output: Optional path to save features
        
        Returns:
            dict: {class_name: {filename: feature_vector}}
        """
        dataset_features = {}
        
        print(f"\n📂 Processing dataset: {dataset_path}")
        
        # Get all class folders
        class_folders = [d for d in os.listdir(dataset_path) 
                        if os.path.isdir(os.path.join(dataset_path, d))]
        
        print(f"Found {len(class_folders)} classes")
        
        for class_name in tqdm(class_folders, desc="Processing classes", unit="class"):
            class_path = os.path.join(dataset_path, class_name)
            class_features = self.extract_from_folder(class_path, save_output=None)
            dataset_features[class_name] = class_features
        
        total_features = sum(len(v) for v in dataset_features.values())
        print(f"✓ Total features extracted: {total_features}")
        
        if save_output:
            self.save_features(dataset_features, save_output)
        
        return dataset_features
    
    def compare(self, feature1, feature2):
        """Compare two features using cosine similarity"""
        return float(np.dot(feature1, feature2))
    
    def save_features(self, features_dict, output_path):
        """Save extracted features to file"""
        with open(output_path, 'wb') as f:
            pickle.dump(features_dict, f)
        print(f"✓ Features saved to: {output_path}")
    
    def load_features(self, input_path):
        """Load features from file"""
        with open(input_path, 'rb') as f:
            features = pickle.load(f)
        print(f"✓ Features loaded from: {input_path}")
        return features


# ===================== TRAINING FUNCTION =====================
def train():
    print("="*60)
    print("IRIS RECOGNITION TRAINING (Simple Mode)")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Feature Dimension: {FEATURE_DIM}D")
    
    # Data transforms
    transform = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    
    # Load dataset
    print(f"\nLoading dataset from: {DATA_DIR}")
    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
    print(f"✓ Total samples: {len(dataset)}")
    print(f"✓ Number of classes: {len(dataset.classes)}")
    
    # Split dataset
    train_size = int(TRAIN_SPLIT * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    print(f"\n✓ Train samples: {train_size}")
    print(f"✓ Test samples: {test_size}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    # Initialize model
    model = MBLNet().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # Save model configuration
    model_config = {
        'img_height': IMG_HEIGHT,
        'img_width': IMG_WIDTH,
        'embed_dim': EMBED_DIM,
        'feature_dim': FEATURE_DIM,
        'batch_size': BATCH_SIZE,
        'learning_rate': LR,
        'margin': MARGIN,
        'epochs': EPOCHS,
        'dataset': DATA_DIR
    }
    
    config_path = os.path.join(OUTPUT_DIR, MODEL_CONFIG_NAME)
    with open(config_path, 'w') as f:
        json.dump(model_config, f, indent=4)
    print(f"✓ Model config saved: {config_path}")
    
    # Training loop
    print(f"\n{'='*60}")
    print("STARTING TRAINING")
    print(f"{'='*60}\n")
    
    best_loss = float('inf')
    training_losses = []
    
    epoch_pbar = tqdm(range(1, EPOCHS + 1), desc="Training", unit="epoch")
    
    for epoch in epoch_pbar:
        model.train()
        total_loss = 0
        
        batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False, unit="batch")
        
        for x, y in batch_pbar:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            embeddings = model(x, return_embedding=True)
            loss = batch_hard_triplet(embeddings, y, MARGIN)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            batch_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(train_loader)
        training_losses.append(avg_loss)
        
        epoch_pbar.set_postfix({
            'avg_loss': f'{avg_loss:.4f}',
            'best': f'{best_loss:.4f}'
        })
        
        if avg_loss < best_loss:
            best_loss = avg_loss
    
    # Save model
    model_path = os.path.join(OUTPUT_DIR, MODEL_NAME)
    torch.save(model.state_dict(), model_path)
    print(f"\n✓ Model weights saved: {model_path}")
    
    # Save complete model (alternative format)
    complete_model_path = os.path.join(OUTPUT_DIR, "complete_model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': model_config,
        'epoch': EPOCHS,
        'best_loss': best_loss
    }, complete_model_path)
    print(f"✓ Complete model checkpoint saved: {complete_model_path}")
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, EPOCHS + 1), training_losses, linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Training Loss', fontsize=12)
    plt.title('Training Loss Over Time', fontsize=14)
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "training_loss.png"), dpi=150)
    plt.close()
    print(f"✓ Training loss plot saved")
    
    # Extract embeddings
    print(f"\n{'='*60}")
    print("EXTRACTING FEATURES")
    print(f"{'='*60}\n")
    
    train_emb, train_lbl = extract_embeddings(model, train_loader)
    test_emb, test_lbl = extract_embeddings(model, test_loader)
    
    print(f"✓ Train features: {train_emb.shape}")
    print(f"✓ Test features: {test_emb.shape}")
    
    # Save embeddings
    emb_path = os.path.join(OUTPUT_DIR, EMBEDDINGS_NAME)
    with open(emb_path, "wb") as f:
        pickle.dump({
            'train_emb': train_emb,
            'train_lbl': train_lbl,
            'test_emb': test_emb,
            'test_lbl': test_lbl,
            'config': model_config
        }, f)
    print(f"✓ Embeddings saved: {emb_path}")
    
    # Evaluate
    print(f"\n{'='*60}")
    print("EVALUATION ON TEST SET")
    print(f"{'='*60}")
    
    results = evaluate(test_emb, test_lbl, "test")
    
    # Save results
    results.update({
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LR,
        'best_training_loss': float(best_loss),
        'final_training_loss': float(training_losses[-1]),
        'dataset': DATA_DIR,
        'train_samples': train_size,
        'test_samples': test_size
    })
    
    results_path = os.path.join(OUTPUT_DIR, RESULTS_NAME)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"\n✓ Results saved: {results_path}")
    print(f"✓ All outputs saved to: {OUTPUT_DIR}")
    print("\n✅ Training Complete!")
    
    return model


# ===================== EXTRACT FEATURES FROM OTHER DATASET =====================
def extract_from_other_dataset(dataset_path, output_name=None):
    """
    Extract features from a different dataset using trained model
    
    Args:
        dataset_path: Path to new dataset
        output_name: Name for output file (default: dataset folder name)
    """
    print("="*60)
    print("FEATURE EXTRACTION FROM NEW DATASET")
    print("="*60)
    
    # Initialize extractor
    extractor = IrisFeatureExtractor()
    
    # Extract features
    if output_name is None:
        output_name = os.path.basename(dataset_path.rstrip('/'))
    
    output_path = os.path.join(OUTPUT_DIR, f"features_{output_name}.pkl")
    
    # Extract from entire dataset
    features = extractor.extract_from_dataset(dataset_path, save_output=output_path)
    
    print(f"\n✅ Feature extraction complete!")
    print(f"Features saved to: {output_path}")
    
    return features


# ===================== MAIN =====================
def main():
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'train':
            # Training mode
            train()
            
        elif command == 'extract':
            # Extract from other dataset
            if len(sys.argv) < 3:
                print("Usage: python script.py extract <dataset_path> [output_name]")
                print("Example: python script.py extract /path/to/CASIA_v1 casia_v1")
                return
            
            dataset_path = sys.argv[2]
            output_name = sys.argv[3] if len(sys.argv) > 3 else None
            extract_from_other_dataset(dataset_path, output_name)
            
        elif command == 'test':
            # Show usage examples
            print("="*60)
            print("USAGE EXAMPLES")
            print("="*60)
            print("\n1. Extract from single image:")
            print("   from script_name import IrisFeatureExtractor")
            print("   extractor = IrisFeatureExtractor()")
            print("   feature = extractor.extract_from_image('image.jpg')")
            
            print("\n2. Extract from folder:")
            print("   features = extractor.extract_from_folder('path/to/folder')")
            
            print("\n3. Extract from entire dataset:")
            print("   python script.py extract /path/to/dataset output_name")
            
            print("\n4. Compare two images:")
            print("   feat1 = extractor.extract_from_image('img1.jpg')")
            print("   feat2 = extractor.extract_from_image('img2.jpg')")
            print("   sim = extractor.compare(feat1, feat2)")
            
    else:
        # Default: training mode
        train()
        
        print("\n" + "="*60)
        print("NEXT STEPS")
        print("="*60)
        print("\n1. Extract from other dataset:")
        print("   python script.py extract /path/to/other/dataset")
        
        print("\n2. Use in your code:")
        print("""
from script_name import IrisFeatureExtractor

extractor = IrisFeatureExtractor()
feature = extractor.extract_from_image('iris.jpg')
        """)


if __name__ == "__main__":
    main()
