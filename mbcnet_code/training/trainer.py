"""Modular training pipeline"""
import os
import json
import pickle
import torch
import torch.optim as optim
from tqdm import tqdm
from utils.visualization import plot_training_loss
from evaluation.evaluator import extract_embeddings, evaluate_model


class Trainer:
    """
    Modular trainer for iris recognition

    Args:
        model: Model to train
        train_loader: Training data loader
        test_loader: Test data loader  
        loss_fn: Loss function
        config: Configuration module
        device: Device to use
    """
    def __init__(self, model, train_loader, test_loader, loss_fn, config, device):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.loss_fn = loss_fn
        self.config = config
        self.device = device

        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config.LEARNING_RATE
        )

        # Training state
        self.current_epoch = 0
        self.training_losses = []
        self.best_loss = float('inf')

        # Create output directory
        os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0

        pbar = tqdm(self.train_loader, 
                   desc=f"Epoch {self.current_epoch + 1}/{self.config.EPOCHS}",
                   leave=False, unit="batch")

        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            embeddings = self.model(images, return_embedding=True)
            loss = self.loss_fn(embeddings, labels)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(self.train_loader)
        self.training_losses.append(avg_loss)

        # Update best loss
        if avg_loss < self.best_loss:
            self.best_loss = avg_loss

        return avg_loss

    def train(self):
        """Full training loop"""
        print("=" * 60)
        print("STARTING TRAINING")
        print("=" * 60)
        print(f"Device: {self.device}")
        print(f"Epochs: {self.config.EPOCHS}")
        print(f"Batch Size: {self.config.BATCH_SIZE}")
        print(f"Learning Rate: {self.config.LEARNING_RATE}")
        print()

        epoch_pbar = tqdm(range(self.config.EPOCHS), desc="Training", unit="epoch")

        for epoch in epoch_pbar:
            self.current_epoch = epoch
            avg_loss = self.train_epoch()

            epoch_pbar.set_postfix({
                'avg_loss': f'{avg_loss:.4f}',
                'best': f'{self.best_loss:.4f}'
            })

        print(f"\n✓ Training complete!")
        print(f"  Final loss: {self.training_losses[-1]:.4f}")
        print(f"  Best loss: {self.best_loss:.4f}")

    def save_model(self):
        """Save model and configuration"""
        model_path = os.path.join(self.config.OUTPUT_DIR, self.config.MODEL_NAME)
        torch.save(self.model.state_dict(), model_path)
        print(f"\n✓ Model saved: {model_path}")

        # Save configuration
        model_config = {
            'img_height': self.config.IMG_HEIGHT,
            'img_width': self.config.IMG_WIDTH,
            'img_channels': self.config.IMG_CHANNELS,
            'backbone_channels': self.config.BACKBONE_CHANNELS,
            'branch_configs': self.config.BRANCH_CONFIGS,
            'embed_dim': self.config.EMBED_DIM,
            'feature_dim': self.config.FEATURE_DIM,
            'attention_type': self.config.ATTENTION_TYPE,
            'attention_kernel': self.config.ATTENTION_KERNEL,
            'loss_type': self.config.LOSS_TYPE,
            'margin': self.config.MARGIN,
            'batch_size': self.config.BATCH_SIZE,
            'learning_rate': self.config.LEARNING_RATE,
            'epochs': self.config.EPOCHS,
            'best_loss': float(self.best_loss),
            'final_loss': float(self.training_losses[-1])
        }

        config_path = os.path.join(self.config.OUTPUT_DIR, 
                                   self.config.MODEL_CONFIG_NAME)
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=4)
        print(f"✓ Config saved: {config_path}")

        # Save complete checkpoint
        checkpoint_path = os.path.join(self.config.OUTPUT_DIR, "checkpoint.pth")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': model_config,
            'epoch': self.config.EPOCHS,
            'training_losses': self.training_losses,
            'best_loss': self.best_loss
        }, checkpoint_path)
        print(f"✓ Checkpoint saved: {checkpoint_path}")

        # Plot training loss
        plot_training_loss(
            self.training_losses,
            os.path.join(self.config.OUTPUT_DIR, "training_loss.png")
        )
        print(f"✓ Training plot saved")

    def evaluate(self):
        """Evaluate on test set"""
        print("\n" + "=" * 60)
        print("EVALUATION")
        print("=" * 60)

        # Extract embeddings
        print("\nExtracting features...")
        train_emb, train_lbl = extract_embeddings(
            self.model, self.train_loader, self.device, return_embedding=False
        )
        test_emb, test_lbl = extract_embeddings(
            self.model, self.test_loader, self.device, return_embedding=False
        )

        print(f"✓ Train features: {train_emb.shape}")
        print(f"✓ Test features: {test_emb.shape}")

        # Save embeddings
        emb_path = os.path.join(self.config.OUTPUT_DIR, self.config.EMBEDDINGS_NAME)
        with open(emb_path, "wb") as f:
            pickle.dump({
                'train_emb': train_emb,
                'train_lbl': train_lbl,
                'test_emb': test_emb,
                'test_lbl': test_lbl
            }, f)
        print(f"✓ Embeddings saved: {emb_path}")

        # Evaluate
        results = evaluate_model(
            test_emb, test_lbl, 
            self.config.OUTPUT_DIR, 
            "test",
            self.config.MAX_PAIRS
        )

        # Add training info
        results.update({
            'best_training_loss': float(self.best_loss),
            'final_training_loss': float(self.training_losses[-1])
        })

        # Save results
        results_path = os.path.join(self.config.OUTPUT_DIR, self.config.RESULTS_NAME)
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"\n✓ Results saved: {results_path}")

        return results