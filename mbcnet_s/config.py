# # ============================================================
# # PATHS
# # ============================================================

# import os

# ROOT_DIR = "/home/nishkal/datasets/iris_db/CASIA_iris_thousand/worldcoin_outputs_images/normalized"
# # casia_iris_thousand_normalized="/home/nishkal/datasets/iris_db/CASIA_iris_thousand/worldcoin_outputs_images/normalized"

# IMAGE_DIR_Train = os.path.join("dataset", "train")
# IMAGE_DIR_Test  = os.path.join("dataset", "recog")

# SAVE_DIR = "output"
# os.makedirs(SAVE_DIR, exist_ok=True)

# CSV_FILE_Train = os.path.join(SAVE_DIR, "iris_train_pairs.csv")
# CSV_FILE_Recog = os.path.join(SAVE_DIR, "iris_test_pairs.csv")


# # ============================================================
# # DATA
# # ============================================================

# IMG_SIZE = 400           # Input size for IRIS images
# BATCH_SIZE = 32
# NUM_WORKERS = 4


# # ============================================================
# # MODEL
# # ============================================================

# IN_CHANNELS = 1          # Grayscale IRIS
# FEATURE_DIM = 256        # Final embedding dimension
# ATTENTION_TYPE = "channel"   # "channel" or "cbam"


# # ============================================================
# # TRAINING
# # ============================================================

# EPOCHS = 50
# LEARNING_RATE = 1e-3
# WEIGHT_DECAY = 1e-3

# # Binary similarity loss
# SIMILARITY_THRESHOLD = 0.5


# # ============================================================
# # ANALYSIS (OPTIONAL)
# # ============================================================

# MARGIN = 1.0   # Used only for distance analysis / ROC / EER


# # ============================================================
# # EXPERIMENT TAGGING
# # ============================================================

# EXPERIMENT_NAME = "IRIS_MBLNet_Siamese"


# ============================================================
# PATHS
# ============================================================

import os

ROOT_DIR = "/home/nishkal/datasets/iris_db/CASIA_v1/worldcoin_outputs_images/normalized"

IMAGE_DIR_Train = os.path.join("dataset", "train")
IMAGE_DIR_Test = os.path.join("dataset", "recog")

SAVE_DIR = "output"
os.makedirs(SAVE_DIR, exist_ok=True)

CSV_FILE_Train = os.path.join(SAVE_DIR, "iris_train_pairs.csv")
CSV_FILE_Recog = os.path.join(SAVE_DIR, "iris_test_pairs.csv")


# ============================================================
# DATA
# ============================================================

IMG_SIZE = 400           # Input size for IRIS images
BATCH_SIZE = 64
NUM_WORKERS = 4


# ============================================================
# MODEL
# ============================================================

IN_CHANNELS = 1          # Grayscale IRIS
FEATURE_DIM = 256        # Final embedding dimension
ATTENTION_TYPE = "channel"   # "channel" or "cbam"


# ============================================================
# TRAINING
# ============================================================

EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3

# Binary similarity loss
SIMILARITY_THRESHOLD = 0.5


# ============================================================
# ANALYSIS (OPTIONAL)
# ============================================================

MARGIN = 1.0   # Used only for distance analysis / ROC / EER


# ============================================================
# EXPERIMENT TAGGING
# ============================================================

EXPERIMENT_NAME = "IRIS_MBLNet_Siamese"
