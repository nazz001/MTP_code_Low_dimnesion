"""Visualization utilities for training and evaluation"""
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc


def plot_roc_curve(fpr, tpr, eer_idx, output_path):
    """Plot ROC curve"""
    roc_auc = auc(fpr, tpr)
    fnr = 1 - tpr
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}", linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.scatter(fpr[eer_idx], tpr[eer_idx], label=f"EER={eer:.4f}", 
                color='red', s=100, zorder=5)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_far_frr(fpr, fnr, thresholds, eer_idx, output_path):
    """Plot FAR/FRR curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, fpr, label="FAR", linewidth=2)
    plt.plot(thresholds, fnr, label="FRR", linewidth=2)
    plt.axvline(thresholds[eer_idx], color='red', linestyle='--', 
                label=f'EER Threshold={thresholds[eer_idx]:.3f}', alpha=0.7)
    plt.xlabel('Threshold', fontsize=12)
    plt.ylabel('Rate', fontsize=12)
    plt.title('FAR/FRR Curves', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(alpha=0.3)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_loss(losses, output_path):
    """Plot training loss over epochs"""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(losses) + 1), losses, linewidth=2, marker='o')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Training Loss', fontsize=12)
    plt.title('Training Loss Over Time', fontsize=14)
    plt.grid(alpha=0.3)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
