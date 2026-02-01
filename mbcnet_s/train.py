# # import os
# # import time
# # import torch
# # from torch.cuda.amp import autocast, GradScaler
# # from tqdm import tqdm
# # import matplotlib.pyplot as plt
# # import config
# # from test import test_model


# # def train_model(
# #     model,
# #     train_loader,
# #     val_loader,
# #     test_loader,
# #     optimizer,
# #     criterion,
# #     device,
# #     scheduler=None,
# #     save_path=None,
# # ):
# #     """
# #     Training loop for IRIS Siamese Verification Model
# #     """

# #     train_losses, val_losses = [], []
# #     train_accuracies, val_accuracies = [], []

# #     best_val_accuracy = 0.0
# #     best_model_path = os.path.join(save_path, "best_model.pth")

# #     model.to(device)
# #     scaler = GradScaler()

# #     for epoch in range(config.EPOCHS):
# #         start_time = time.time()
# #         model.train()

# #         running_loss = 0.0
# #         correct_train = 0
# #         total_train = 0

# #         progress_bar = tqdm(
# #             train_loader,
# #             desc=f"Epoch {epoch + 1}/{config.EPOCHS}",
# #             ncols=100
# #         )

# #         for img1, _, img2, _, pair_label in progress_bar:
# #             img1 = img1.to(device, non_blocking=True)
# #             img2 = img2.to(device, non_blocking=True)
# #             pair_label = pair_label.to(device, non_blocking=True)

# #             optimizer.zero_grad()

# #             with autocast(device_type="cuda"):
# #                 outputs = model(img1, img2)
# #                 loss_dict = criterion(
# #                     outputs,
# #                     {"pair_label": pair_label}
# #                 )
# #                 loss = loss_dict["total_loss"]

# #             scaler.scale(loss).backward()
# #             scaler.step(optimizer)
# #             scaler.update()

# #             running_loss += loss.item()

# #             # Accuracy
# #             similarity_score = outputs["similarity_score"].squeeze(1)
# #             predictions = (torch.sigmoid(similarity_score) >= 0.5).float()
# #             correct_train += (predictions == pair_label).sum().item()
# #             total_train += pair_label.size(0)

# #         train_loss = running_loss / len(train_loader)
# #         train_accuracy = correct_train / total_train

# #         train_losses.append(train_loss)
# #         train_accuracies.append(train_accuracy)

# #         # ---------------- VALIDATION ----------------
# #         val_loss, val_accuracy, val_acc_1, val_acc_0 = test_model(
# #             model, criterion, val_loader, device
# #         )

# #         val_losses.append(val_loss)
# #         val_accuracies.append(val_accuracy)

# #         if scheduler:
# #             scheduler.step()

# #         epoch_time = time.time() - start_time
# #         minutes, seconds = divmod(epoch_time, 60)

# #         print(
# #             f"Epoch [{epoch + 1}/{config.EPOCHS}] | "
# #             f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f} | "
# #             f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f} "
# #             f"(Genuine: {val_acc_1:.4f}, Imposter: {val_acc_0:.4f}) | "
# #             f"Time: {int(minutes)}m {int(seconds)}s",
# #             flush=True
# #         )

# #         # Save best model
# #         if val_accuracy > best_val_accuracy:
# #             best_val_accuracy = val_accuracy
# #             torch.save(model.state_dict(), best_model_path)
# #             print(
# #                 f"✓ Saved best model @ {best_model_path} "
# #                 f"(Val Acc = {best_val_accuracy:.4f})",
# #                 flush=True
# #             )

# #         torch.cuda.empty_cache()

# #         # ---------------- PERIODIC TEST ----------------
# #         if (epoch + 1) % 10 == 0:
# #             test_loss, test_acc, test_acc_1, test_acc_0 = test_model(
# #                 model, criterion, test_loader, device
# #             )
# #             print(
# #                 f"***** TEST | Acc: {test_acc:.4f}, Loss: {test_loss:.4f} "
# #                 f"(Genuine: {test_acc_1:.4f}, Imposter: {test_acc_0:.4f}) *****",
# #                 flush=True
# #             )

# #     # ---------------- SAVE CURVES ----------------
# #     if save_path:
# #         os.makedirs(save_path, exist_ok=True)

# #         plt.figure()
# #         plt.plot(train_losses, label="Train Loss")
# #         plt.plot(val_losses, label="Val Loss")
# #         plt.legend()
# #         plt.savefig(os.path.join(save_path, "loss_curve.png"))
# #         plt.close()

# #         plt.figure()
# #         plt.plot(train_accuracies, label="Train Acc")
# #         plt.plot(val_accuracies, label="Val Acc")
# #         plt.legend()
# #         plt.savefig(os.path.join(save_path, "accuracy_curve.png"))
# #         plt.close()

# #         print("✓ Loss & accuracy curves saved")

# #     return best_model_path, train_losses, val_losses, train_accuracies, val_accuracies



# import os
# import time
# import torch
# # from torch.cuda.amp import autocast, GradScaler
# from torch.amp import autocast, GradScaler

# from tqdm import tqdm
# import matplotlib.pyplot as plt

# import config
# from test import test_model


# def train_model(
#     model,
#     train_loader,
#     val_loader,
#     test_loader,
#     optimizer,
#     criterion,
#     device,
#     scheduler=None,
#     save_path=None
# ):
#     """
#     Training loop for IRIS Siamese Verification Model
#     (tqdm + AMP + best-model saving)
#     """

#     train_losses, val_losses = [], []
#     train_accuracies, val_accuracies = [], []

#     best_val_accuracy = 0.0
#     best_model_path = None

#     model.to(device)
#     scaler = GradScaler()

#     for epoch in range(config.EPOCHS):
#         start_time = time.time()
#         model.train()

#         running_loss = 0.0
#         correct_train = 0
#         total_train = 0

#         progress_bar = tqdm(
#             train_loader,
#             desc=f"Epoch {epoch + 1}/{config.EPOCHS} [Train]",
#             ncols=100
#         )

#         # ===================== TRAIN =====================
#         for img1, _, img2, _, pair_label in progress_bar:
#             img1 = img1.to(device, non_blocking=True)
#             img2 = img2.to(device, non_blocking=True)
#             pair_label = pair_label.to(device, non_blocking=True)

#             optimizer.zero_grad()

#             with autocast("cuda"):
#                 outputs = model(img1, img2)
#                 loss_dict = criterion(
#                     outputs,
#                     {"pair_label": pair_label}
#                 )
#                 loss = loss_dict["total_loss"]

#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()

#             running_loss += loss.item()

#             similarity_score = outputs["similarity_score"].squeeze(1)
#             predictions = (torch.sigmoid(similarity_score) >= 0.5).float()

#             correct_train += (predictions == pair_label).sum().item()
#             total_train += pair_label.size(0)

#             progress_bar.set_postfix(
#                 loss=f"{loss.item():.4f}",
#                 acc=f"{correct_train / total_train:.4f}"
#             )

#         train_loss = running_loss / len(train_loader)
#         train_accuracy = correct_train / total_train

#         train_losses.append(train_loss)
#         train_accuracies.append(train_accuracy)

#         # ===================== VALIDATION =====================
#         val_loss, val_accuracy, val_acc_1, val_acc_0 = test_model(
#             model, criterion, val_loader, device
#         )

#         val_losses.append(val_loss)
#         val_accuracies.append(val_accuracy)

#         if scheduler is not None:
#             scheduler.step()

#         epoch_time = time.time() - start_time
#         minutes, seconds = divmod(epoch_time, 60)

#         print(
#             f"Epoch [{epoch + 1}/{config.EPOCHS}] | "
#             f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f} | "
#             f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f} "
#             f"(Genuine: {val_acc_1:.4f}, Imposter: {val_acc_0:.4f}) | "
#             f"Time: {int(minutes)}m {int(seconds)}s",
#             flush=True
#         )

#         # ===================== SAVE BEST =====================
#         if save_path and val_accuracy > best_val_accuracy:
#             best_val_accuracy = val_accuracy
#             os.makedirs(save_path, exist_ok=True)
#             best_model_path = os.path.join(save_path, "best_model.pth")
#             torch.save(model.state_dict(), best_model_path)
#             print(
#                 f"✓ Saved best model @ {best_model_path} "
#                 f"(Val Acc = {best_val_accuracy:.4f})",
#                 flush=True
#             )

#         torch.cuda.empty_cache()

#         # ===================== PERIODIC TEST =====================
#         if (epoch + 1) % 10 == 0:
#             test_loss, test_acc, test_acc_1, test_acc_0 = test_model(
#                 model, criterion, test_loader, device
#             )
#             print(
#                 f"***** TEST | Acc: {test_acc:.4f}, Loss: {test_loss:.4f} "
#                 f"(Genuine: {test_acc_1:.4f}, Imposter: {test_acc_0:.4f}) *****",
#                 flush=True
#             )

#     # ===================== SAVE CURVES =====================
#     if save_path:
#         plt.figure()
#         plt.plot(train_losses, label="Train Loss")
#         plt.plot(val_losses, label="Val Loss")
#         plt.legend()
#         plt.savefig(os.path.join(save_path, "loss_curve.png"))
#         plt.close()

#         plt.figure()
#         plt.plot(train_accuracies, label="Train Acc")
#         plt.plot(val_accuracies, label="Val Acc")
#         plt.legend()
#         plt.savefig(os.path.join(save_path, "accuracy_curve.png"))
#         plt.close()

#         print("✓ Loss & accuracy curves saved")

#     return (
#         best_model_path,
#         train_losses,
#         val_losses,
#         train_accuracies,
#         val_accuracies
#     )


import os
import torch
from tqdm import tqdm
from utils import calculate_accuracy


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    """
    Train model for one epoch.
    
    Args:
        model: Neural network model
        train_loader: DataLoader for training data
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        
    Returns:
        avg_loss, avg_accuracy
    """
    model.train()
    
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0
    
    progress_bar = tqdm(train_loader, desc="Training", leave=False)
    
    for batch in progress_bar:
        img1, class1, img2, class2, pair_label = batch
        
        # Move to device
        img1 = img1.to(device)
        img2 = img2.to(device)
        pair_label = pair_label.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(img1, img2)
        
        # Calculate loss
        targets = {"pair_label": pair_label}
        loss_dict = criterion(outputs, targets)
        loss = loss_dict["total_loss"]
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        similarity_score = outputs["similarity_score"].squeeze(1)
        accuracy = calculate_accuracy(similarity_score, pair_label)
        
        # Accumulate metrics
        total_loss += loss.item()
        total_accuracy += accuracy
        num_batches += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{accuracy:.4f}'
        })
    
    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    
    return avg_loss, avg_accuracy


def validate(model, val_loader, criterion, device):
    """
    Validate model on validation set.
    
    Args:
        model: Neural network model
        val_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to validate on
        
    Returns:
        avg_loss, avg_accuracy
    """
    model.eval()
    
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validation", leave=False)
        
        for batch in progress_bar:
            img1, class1, img2, class2, pair_label = batch
            
            # Move to device
            img1 = img1.to(device)
            img2 = img2.to(device)
            pair_label = pair_label.to(device)
            
            # Forward pass
            outputs = model(img1, img2)
            
            # Calculate loss
            targets = {"pair_label": pair_label}
            loss_dict = criterion(outputs, targets)
            loss = loss_dict["total_loss"]
            
            # Calculate accuracy
            similarity_score = outputs["similarity_score"].squeeze(1)
            accuracy = calculate_accuracy(similarity_score, pair_label)
            
            # Accumulate metrics
            total_loss += loss.item()
            total_accuracy += accuracy
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{accuracy:.4f}'
            })
    
    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    
    return avg_loss, avg_accuracy


def train_model(
    model,
    train_loader,
    val_loader,
    test_loader,
    optimizer,
    criterion,
    device,
    scheduler,
    save_path,
    num_epochs=50
):
    """
    Complete training loop with validation and testing.
    
    Args:
        model: Neural network model
        train_loader: Training data loader
        val_loader: Validation data loader (recognition set)
        test_loader: Test data loader (within-split test)
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        scheduler: Learning rate scheduler
        save_path: Directory to save checkpoints
        num_epochs: Number of epochs to train
        
    Returns:
        best_model_path, train_losses, val_losses, train_accs, val_accs, best_test_acc
    """
    
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    best_val_acc = 0.0
    best_test_acc = 0.0
    best_model_path = os.path.join(save_path, "best_model.pth")
    
    print(f"\nStarting training for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*60}")
        
        # Training
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation (recognition set)
        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Test (within-split)
        test_loss, test_acc = validate(
            model, test_loader, criterion, device
        )
        
        # Learning rate step
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print epoch summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")
        print(f"  Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.4f}")
        print(f"  LR: {current_lr:.6f}")
        
        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✓ Best model saved! Val Acc: {val_acc:.4f}")
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Best Val Acc: {best_val_acc:.4f}")
    print(f"Best Test Acc: {best_test_acc:.4f}")
    print(f"{'='*60}\n")
    
    return best_model_path, train_losses, val_losses, train_accs, val_accs, best_test_acc