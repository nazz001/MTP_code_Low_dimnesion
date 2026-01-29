"""
Configuration file for Iris Recognition System
Modify parameters here to customize the system
"""
import torch

# ===================== DATASET PATHS =====================
CASIA_IRIS_THOUSAND = "/home/nishkal/datasets/iris_db/CASIA_iris_thousand/worldcoin_outputs_images/normalized"
#CASIA_V1 = "/home/nishkal/datasets/iris_db/CASIA_v1"
#IITD_V1 = "/home/nishkal/datasets/iris_db/IITD_v1"

# Select active dataset
DATA_DIR = CASIA_IRIS_THOUSAND


# ===================== OUTPUT SETTINGS =====================
OUTPUT_DIR = "outputs_simple"
MODEL_NAME = "iris_model.pth"
MODEL_CONFIG_NAME = "model_config.json"
EMBEDDINGS_NAME = "embeddings.pkl"
RESULTS_NAME = "results.json"


# ===================== IMAGE SETTINGS =====================
IMG_HEIGHT = 60
IMG_WIDTH = 240
IMG_CHANNELS = 1  # Grayscale


# ===================== TRAINING HYPERPARAMETERS =====================
SEED = 42
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-4
TRAIN_SPLIT = 0.8  # 80% train, 20% test


# ===================== MODEL ARCHITECTURE =====================
# Backbone channels for each stage
BACKBONE_CHANNELS = [16, 24, 32, 48, 64]

# Branch configurations: (kernel_size, dilation)
BRANCH_CONFIGS = [
    (3, 1),  # Branch 1: 3x3 kernel, dilation=1
    (5, 1),  # Branch 2: 5x5 kernel, dilation=1
    (3, 2),  # Branch 3: 3x3 kernel, dilation=2
]

# Embedding and feature dimensions
EMBED_DIM = 256
FEATURE_DIM = 128

# Attention mechanism
ATTENTION_TYPE = "channel"  # Options: "channel", "spatial", "cbam", "none"
ATTENTION_KERNEL = 5


# ===================== LOSS FUNCTION =====================
LOSS_TYPE = "triplet"  # Options: "triplet", "contrastive", "arcface"
MARGIN = 0.3
TRIPLET_MODE = "batch_hard"  # Options: "batch_hard", "batch_all"


# ===================== EVALUATION =====================
MAX_PAIRS = 200000  # Maximum pairs for evaluation


# ===================== DEVICE =====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===================== DATA AUGMENTATION =====================
USE_AUGMENTATION = False
AUGMENTATION_CONFIG = {
    "random_horizontal_flip": 0.5,
    "random_rotation": 5,
    "random_brightness": 0.1,
    "random_contrast": 0.1,
}
