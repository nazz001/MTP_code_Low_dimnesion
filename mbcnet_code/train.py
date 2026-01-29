"""Main training script"""
import sys
import os
import config.config as cfg
from models.model import build_model
from models.losses import get_loss_function
from data.dataset import load_dataset
from training.trainer import Trainer
from utils.seed import set_seed


def main():
    print("=" * 60)
    print("IRIS RECOGNITION TRAINING")
    print("=" * 60)

    # Set seed
    set_seed(cfg.SEED)

    # Load dataset
    print(f"\nLoading dataset: {cfg.DATA_DIR}")
    train_loader, test_loader, num_classes, train_size, test_size = load_dataset(
        cfg.DATA_DIR,
        cfg.IMG_HEIGHT,
        cfg.IMG_WIDTH,
        cfg.TRAIN_SPLIT,
        cfg.BATCH_SIZE,
        cfg.USE_AUGMENTATION,
        cfg.AUGMENTATION_CONFIG if cfg.USE_AUGMENTATION else None
    )

    print(f"✓ Classes: {num_classes}")
    print(f"✓ Train samples: {train_size}")
    print(f"✓ Test samples: {test_size}")

    # Build model
    print(f"\nBuilding model...")
    model = build_model(cfg)
    print(f"✓ Model architecture: MBLNet")
    print(f"✓ Branches: {len(cfg.BRANCH_CONFIGS)}")
    print(f"✓ Attention: {cfg.ATTENTION_TYPE}")
    print(f"✓ Feature dim: {cfg.FEATURE_DIM}D")

    # Get loss function
    if cfg.LOSS_TYPE == "triplet":
        loss_fn = get_loss_function("triplet", margin=cfg.MARGIN, mode=cfg.TRIPLET_MODE)
    elif cfg.LOSS_TYPE == "contrastive":
        loss_fn = get_loss_function("contrastive", margin=cfg.MARGIN)
    elif cfg.LOSS_TYPE == "arcface":
        loss_fn = get_loss_function("arcface", in_features=cfg.EMBED_DIM, 
                                   out_features=num_classes)
    else:
        raise ValueError(f"Unknown loss: {cfg.LOSS_TYPE}")

    print(f"✓ Loss function: {cfg.LOSS_TYPE}")

    # Create trainer
    trainer = Trainer(model, train_loader, test_loader, loss_fn, cfg, cfg.DEVICE)

    # Train
    trainer.train()

    # Save
    trainer.save_model()

    # Evaluate
    results = trainer.evaluate()

    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"AUC: {results['auc']:.4f}")
    print(f"EER: {results['eer']:.4f}")
    print(f"\nOutputs saved to: {cfg.OUTPUT_DIR}")


if __name__ == "__main__":
    main()