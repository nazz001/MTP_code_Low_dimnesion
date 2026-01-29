"""Model evaluation utilities"""
import os
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from utils.visualization import plot_roc_curve, plot_far_frr


def extract_embeddings(model, data_loader, device, return_embedding=False):
    """
    Extract embeddings from model

    Args:
        model: Trained model
        data_loader: DataLoader
        device: Device to use
        return_embedding: If True, return 256D embeddings, else 128D features

    Returns:
        embeddings: numpy array of embeddings
        labels: numpy array of labels
    """
    model.eval()
    embeddings_list = []
    labels_list = []

    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Extracting embeddings", 
                                   leave=False, unit="batch"):
            images = images.to(device)
            emb = model(images, return_embedding=return_embedding).cpu().numpy()
            embeddings_list.append(emb)
            labels_list.append(labels.numpy())

    embeddings = np.vstack(embeddings_list)
    labels = np.concatenate(labels_list)

    return embeddings, labels


def evaluate_model(embeddings, labels, output_dir, save_prefix="test", 
                   max_pairs=200000):
    """
    Evaluate model using embeddings

    Args:
        embeddings: Feature vectors
        labels: Corresponding labels
        output_dir: Directory to save results
        save_prefix: Prefix for saved files
        max_pairs: Maximum number of pairs to evaluate

    Returns:
        results: Dictionary with AUC, EER, threshold
    """
    similarities = []
    y_true = []
    count = 0
    n = len(embeddings)

    total_pairs = min(max_pairs, n * (n - 1) // 2)

    print(f"\nEvaluating {total_pairs:,} pairs...")

    with tqdm(total=total_pairs, desc="Computing similarities", unit="pairs") as pbar:
        for i in range(n):
            for j in range(i + 1, n):
                sim = np.dot(embeddings[i], embeddings[j])
                similarities.append(sim)
                y_true.append(int(labels[i] == labels[j]))
                count += 1
                pbar.update(1)

                if count >= max_pairs:
                    break
            if count >= max_pairs:
                break

    similarities = np.array(similarities)
    y_true = np.array(y_true)

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, similarities)
    roc_auc = auc(fpr, tpr)

    # Compute EER
    fnr = 1 - tpr
    eer_idx = np.argmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2

    print(f"\n📊 Results: AUC={roc_auc:.4f}, EER={eer:.4f}")

    # Save plots
    os.makedirs(output_dir, exist_ok=True)
    plot_roc_curve(fpr, tpr, eer_idx, 
                   os.path.join(output_dir, f"roc_{save_prefix}.png"))
    plot_far_frr(fpr, fnr, thresholds, eer_idx, 
                 os.path.join(output_dir, f"far_frr_{save_prefix}.png"))

    results = {
        "auc": float(roc_auc),
        "eer": float(eer),
        "eer_threshold": float(thresholds[eer_idx]),
        "num_pairs_evaluated": count
    }

    return results