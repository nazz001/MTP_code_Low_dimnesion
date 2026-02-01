# import os
# import csv
# import torch
# import pandas as pd
# import matplotlib.pyplot as plt
# from datetime import datetime
# from sklearn.model_selection import train_test_split
# import config


# # ============================================================
# # METRIC SAVING
# # ============================================================

# def save_metrics_to_csv(
#     train_losses,
#     val_losses,
#     train_accuracies,
#     val_accuracies,
#     save_path
# ):
#     csv_path = os.path.join(save_path, "training_metrics.csv")

#     with open(csv_path, "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow([
#             "Epoch",
#             "Train Loss",
#             "Val Loss",
#             "Train Accuracy",
#             "Val Accuracy"
#         ])

#         for epoch, (tl, vl, ta, va) in enumerate(
#             zip(train_losses, val_losses, train_accuracies, val_accuracies),
#             start=1
#         ):
#             writer.writerow([epoch, tl, vl, ta, va])

#     print(f"✓ Metrics saved to {csv_path}")


# # ============================================================
# # CSV HANDLING
# # ============================================================

# def combine_csv_files(csv_file_1, csv_file_2, output_csv):
#     df1 = pd.read_csv(csv_file_1)
#     df2 = pd.read_csv(csv_file_2)

#     combined_df = pd.concat([df1, df2], ignore_index=True)
#     combined_df.to_csv(output_csv, index=False)

#     print(f"✓ Combined CSV saved to {output_csv}")


# def convert_class_labels(csv_path):
#     """
#     Converts string identity labels to integer labels.
#     (Only needed if identity info is required later)
#     """
#     df = pd.read_csv(csv_path)

#     unique_classes = pd.concat([df["class1"], df["class2"]]).unique()
#     mapping = {cls: idx for idx, cls in enumerate(unique_classes)}

#     df["class1"] = df["class1"].map(mapping)
#     df["class2"] = df["class2"].map(mapping)

#     df.to_csv(csv_path, index=False)
#     return mapping


# # ============================================================
# # DATA BALANCING (IMPORTANT FOR IRIS)
# # ============================================================

# def generate_equal_label_dataframes(csv_path):
#     """
#     Ensures equal number of genuine and imposter pairs.
#     """
#     df = pd.read_csv(csv_path)
#     df = df.sample(frac=1).reset_index(drop=True)

#     pos_df = df[df["label"] == 1]
#     neg_df = df[df["label"] == 0]

#     num_pos = len(pos_df)
#     num_neg = len(neg_df)

#     num_splits = (num_neg + num_pos - 1) // num_pos

#     balanced_dfs = []

#     for i in range(num_splits):
#         neg_slice = neg_df.iloc[i * num_pos:(i + 1) * num_pos]
#         combined = pd.concat([pos_df, neg_slice], ignore_index=True)
#         balanced_dfs.append(combined)

#     num_classes = df["class1"].nunique()
#     return balanced_dfs, num_classes


# # ============================================================
# # TRAIN / TEST SPLIT
# # ============================================================

# def split_dataset(df, test_size=0.2, random_state=42):
#     df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

#     train_df, test_df = train_test_split(
#         df,
#         test_size=test_size,
#         stratify=df["label"],
#         random_state=random_state
#     )

#     return train_df, test_df


# # ============================================================
# # ACCURACY (IRIS-SPECIFIC)
# # ============================================================

# def calculate_accuracy(similarity_logits, pair_labels, threshold=0.5):
#     """
#     Computes accuracy from similarity logits.
#     """
#     probs = torch.sigmoid(similarity_logits)
#     predictions = (probs >= threshold).float()
#     correct = (predictions == pair_labels).sum().item()
#     return correct / pair_labels.size(0)


# # ============================================================
# # CONFIG LOGGING
# # ============================================================

# def save_config_to_csv(best_test_accuracy, filename="config_log.csv"):
#     config_data = {
#         k: v for k, v in vars(config).items()
#         if not k.startswith("__")
#     }

#     config_data["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     config_data["Best_Test_Accuracy"] = best_test_accuracy

#     if os.path.exists(filename):
#         df = pd.read_csv(filename)
#         for key in config_data:
#             if key not in df.columns:
#                 df[key] = None
#     else:
#         df = pd.DataFrame(columns=config_data.keys())

#     df = pd.concat([df, pd.DataFrame([config_data])], ignore_index=True)
#     df.to_csv(filename, index=False)

#     print(f"✓ Config saved to {filename}")


# # ============================================================
# # DISTANCE DISTRIBUTION (EER READY)
# # ============================================================

# def plot_distance_distribution(pos_scores, neg_scores, save_path="."):
#     plt.figure(figsize=(10, 5))
#     plt.hist(pos_scores, bins=30, alpha=0.6, label="Genuine")
#     plt.hist(neg_scores, bins=30, alpha=0.6, label="Imposter")
#     plt.xlabel("Similarity Score")
#     plt.ylabel("Frequency")
#     plt.title("Genuine vs Imposter Score Distribution")
#     plt.legend()

#     path = os.path.join(save_path, "score_distribution.png")
#     plt.savefig(path)
#     plt.close()

#     print(f"✓ Score distribution saved to {path}")


import os
import csv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def split_dataset(df, test_size=0.2, random_state=42):
    """
    Split a DataFrame into train and test sets.
    
    Args:
        df (pd.DataFrame): Input DataFrame
        test_size (float): Fraction for test set
        random_state (int): Random seed
        
    Returns:
        train_df, test_df
    """
    train_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
        shuffle=True
    )
    
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def convert_class_labels(csv_file):
    """
    Convert string class labels to integer labels (0-indexed).
    
    Modifies the CSV file in-place.
    
    Args:
        csv_file (str): Path to CSV file
    """
    df = pd.read_csv(csv_file)
    
    # Get unique classes
    if 'class1' in df.columns:
        # Pair CSV
        unique_classes = sorted(set(df['class1'].unique()) | set(df['class2'].unique()))
        class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
        
        df['class1'] = df['class1'].map(class_to_idx)
        df['class2'] = df['class2'].map(class_to_idx)
    else:
        # Single image CSV
        unique_classes = sorted(df['class_label'].unique())
        class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}
        
        df['class_label'] = df['class_label'].map(class_to_idx)
    
    df.to_csv(csv_file, index=False)
    print(f"✓ Converted class labels to integers: {len(unique_classes)} classes")


def generate_equal_label_dataframes(csv_file, max_splits=10):
    """
    Generate balanced train/test splits from a pairs CSV.
    
    Strategy: Create multiple balanced subsets where genuine and 
    imposter pairs are roughly equal.
    
    Args:
        csv_file (str): Path to pairs CSV
        max_splits (int): Maximum number of balanced splits to create
        
    Returns:
        list of DataFrames, number of classes
    """
    df = pd.read_csv(csv_file)
    
    # Separate genuine and imposter pairs
    genuine_pairs = df[df['label'] == 1].reset_index(drop=True)
    imposter_pairs = df[df['label'] == 0].reset_index(drop=True)
    
    print(f"✓ Genuine pairs: {len(genuine_pairs)}")
    print(f"✓ Imposter pairs: {len(imposter_pairs)}")
    
    # Determine split size
    min_count = min(len(genuine_pairs), len(imposter_pairs))
    split_size = min_count // max_splits
    
    if split_size < 10:
        print(f"⚠ Warning: Very small split size ({split_size}). Using single split.")
        max_splits = 1
        split_size = min_count
    
    dataframe_list = []
    
    # Shuffle both
    genuine_pairs = genuine_pairs.sample(frac=1, random_state=42).reset_index(drop=True)
    imposter_pairs = imposter_pairs.sample(frac=1, random_state=42).reset_index(drop=True)
    
    for i in range(max_splits):
        start_idx = i * split_size
        end_idx = start_idx + split_size
        
        # Don't exceed available data
        if end_idx > min_count:
            break
            
        genuine_subset = genuine_pairs.iloc[start_idx:end_idx]
        imposter_subset = imposter_pairs.iloc[start_idx:end_idx]
        
        # Combine and shuffle
        balanced_df = pd.concat([genuine_subset, imposter_subset], ignore_index=True)
        balanced_df = balanced_df.sample(frac=1, random_state=42 + i).reset_index(drop=True)
        
        dataframe_list.append(balanced_df)
    
    # Get number of unique classes
    num_classes = len(set(df['class1'].unique()) | set(df['class2'].unique()))
    
    print(f"✓ Created {len(dataframe_list)} balanced splits")
    print(f"✓ Each split contains ~{split_size * 2} pairs")
    
    return dataframe_list, num_classes


def save_metrics_to_csv(train_losses, val_losses, train_accs, val_accs, save_dir):
    """
    Save training metrics to CSV file.
    
    Args:
        train_losses (list): Training losses per epoch
        val_losses (list): Validation losses per epoch
        train_accs (list): Training accuracies per epoch
        val_accs (list): Validation accuracies per epoch
        save_dir (str): Directory to save CSV
    """
    metrics_file = os.path.join(save_dir, "training_metrics.csv")
    
    with open(metrics_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc'])
        
        for epoch in range(len(train_losses)):
            writer.writerow([
                epoch + 1,
                train_losses[epoch],
                val_losses[epoch],
                train_accs[epoch],
                val_accs[epoch]
            ])
    
    print(f"✓ Metrics saved to: {metrics_file}")


def calculate_accuracy(predictions, labels, threshold=0.5):
    """
    Calculate binary classification accuracy.
    
    Args:
        predictions (torch.Tensor): Predicted similarity scores (logits)
        labels (torch.Tensor): Ground truth labels (0 or 1)
        threshold (float): Decision threshold (applied after sigmoid)
        
    Returns:
        float: Accuracy (0-1)
    """
    import torch
    
    # Apply sigmoid to convert logits to probabilities
    probs = torch.sigmoid(predictions)
    
    # Apply threshold
    pred_labels = (probs > threshold).float()
    
    # Calculate accuracy
    correct = (pred_labels == labels).float().sum()
    accuracy = correct / len(labels)
    
    return accuracy.item()