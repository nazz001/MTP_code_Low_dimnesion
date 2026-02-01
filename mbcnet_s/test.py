import torch


def test_model(model, criterion, test_loader, device):
    """
    Test function for IRIS Siamese Verification Model

    Metrics:
    - Average test loss
    - Overall accuracy
    - Accuracy for genuine pairs (label = 1)
    - Accuracy for imposter pairs (label = 0)
    """

    model.eval()
    test_loss = 0.0

    # Accuracy counters
    correct_total = 0
    total_samples = 0

    correct_genuine = 0
    total_genuine = 0

    correct_imposter = 0
    total_imposter = 0

    with torch.no_grad():
        for img1, _, img2, _, pair_label in test_loader:
            # Move to device
            img1 = img1.to(device)
            img2 = img2.to(device)
            pair_label = pair_label.to(device)

            # Forward pass
            outputs = model(img1, img2)

            # Compute loss
            loss_dict = criterion(
                outputs,
                {"pair_label": pair_label}
            )

            loss = loss_dict["total_loss"]
            test_loss += loss.item()

            # Similarity prediction
            similarity_score = outputs["similarity_score"].squeeze(1)
            predictions = (torch.sigmoid(similarity_score) >= 0.5).float()

            # Overall accuracy
            correct_total += (predictions == pair_label).sum().item()
            total_samples += pair_label.size(0)

            # Genuine / Imposter accuracy
            genuine_mask = pair_label == 1
            imposter_mask = pair_label == 0

            if genuine_mask.any():
                correct_genuine += (predictions[genuine_mask] == 1).sum().item()
                total_genuine += genuine_mask.sum().item()

            if imposter_mask.any():
                correct_imposter += (predictions[imposter_mask] == 0).sum().item()
                total_imposter += imposter_mask.sum().item()

    # Final metrics
    avg_test_loss = test_loss / len(test_loader)

    overall_accuracy = (
        correct_total / total_samples if total_samples > 0 else 0
    )

    genuine_accuracy = (
        correct_genuine / total_genuine if total_genuine > 0 else 0
    )

    imposter_accuracy = (
        correct_imposter / total_imposter if total_imposter > 0 else 0
    )

    return (
        avg_test_loss,
        overall_accuracy,
        genuine_accuracy,
        imposter_accuracy
    )
