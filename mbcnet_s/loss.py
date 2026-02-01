# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class IrisSiameseLoss(nn.Module):
#     """
#     Loss function for IRIS Siamese Verification Model

#     Current version supports:
#     - Similarity loss (primary)
#     - Contrastive loss (metric learning)

#     Future-ready hooks:
#     - Reconstruction loss (AE)
#     - GAN losses (Generator / Discriminator)

#     This design follows the philosophy of EnhancedAELoss,
#     but is SAFE for the current IRIS MBLNet architecture.
#     """

#     def __init__(
#         self,
#         similarity_weight=1.0,
#         contrastive_weight=1.0,
#         margin=1.0
#     ):
#         super().__init__()

#         self.similarity_weight = similarity_weight
#         self.contrastive_weight = contrastive_weight
#         self.margin = margin

#     def forward(self, outputs, targets):
#         """
#         Args:
#             outputs (dict): Model outputs
#                 Required keys (current):
#                     - 'similarity_score': Tensor [B, 1]
#                     - 'features': (f1, f2)

#             targets (dict):
#                 - 'pair_label': Tensor [B] (1 = genuine, 0 = imposter)

#         Returns:
#             dict of losses
#         """

#         # --------------------------------------------------
#         # Extract inputs
#         # --------------------------------------------------
#         similarity_score = outputs["similarity_score"].squeeze(1)
#         f1, f2 = outputs["features"]
#         pair_label = targets["pair_label"].float()

#         # --------------------------------------------------
#         # Similarity Loss (Binary Verification)
#         # --------------------------------------------------
#         similarity_loss = F.binary_cross_entropy_with_logits(
#             similarity_score,
#             pair_label
#         )

#         # --------------------------------------------------
#         # Contrastive Loss (Metric Learning)
#         # --------------------------------------------------
#         f1 = F.normalize(f1, p=2, dim=1)
#         f2 = F.normalize(f2, p=2, dim=1)

#         distances = F.pairwise_distance(f1, f2)

#         contrastive_loss = (
#             pair_label * distances.pow(2) +
#             (1 - pair_label) * F.relu(self.margin - distances).pow(2)
#         ).mean()

#         # --------------------------------------------------
#         # Total Loss
#         # --------------------------------------------------
#         total_loss = (
#             self.similarity_weight * similarity_loss +
#             self.contrastive_weight * contrastive_loss
#         )

#         return {
#             "total_loss": total_loss,
#             "similarity_loss": similarity_loss,
#             "contrastive_loss": contrastive_loss
#         }


import torch
import torch.nn as nn
import torch.nn.functional as F


class IrisSiameseLoss(nn.Module):
    """
    Loss function for IRIS Siamese Verification Model

    Current version supports:
    - Similarity loss (primary)
    - Contrastive loss (metric learning)

    Future-ready hooks:
    - Reconstruction loss (AE)
    - GAN losses (Generator / Discriminator)

    This design follows the philosophy of EnhancedAELoss,
    but is SAFE for the current IRIS MBLNet architecture.
    """

    def __init__(
        self,
        similarity_weight=1.0,
        contrastive_weight=1.0,
        margin=1.0
    ):
        super().__init__()

        self.similarity_weight = similarity_weight
        self.contrastive_weight = contrastive_weight
        self.margin = margin

    def forward(self, outputs, targets):
        """
        Args:
            outputs (dict): Model outputs
                Required keys (current):
                    - 'similarity_score': Tensor [B, 1]
                    - 'features': (f1, f2)

            targets (dict):
                - 'pair_label': Tensor [B] (1 = genuine, 0 = imposter)

        Returns:
            dict of losses
        """

        # --------------------------------------------------
        # Extract inputs
        # --------------------------------------------------
        similarity_score = outputs["similarity_score"].squeeze(1)
        f1, f2 = outputs["features"]
        pair_label = targets["pair_label"].float()

        # --------------------------------------------------
        # Similarity Loss (Binary Verification)
        # --------------------------------------------------
        similarity_loss = F.binary_cross_entropy_with_logits(
            similarity_score,
            pair_label
        )

        # --------------------------------------------------
        # Contrastive Loss (Metric Learning)
        # --------------------------------------------------
        f1 = F.normalize(f1, p=2, dim=1)
        f2 = F.normalize(f2, p=2, dim=1)

        distances = F.pairwise_distance(f1, f2)

        contrastive_loss = (
            pair_label * distances.pow(2) +
            (1 - pair_label) * F.relu(self.margin - distances).pow(2)
        ).mean()

        # --------------------------------------------------
        # Total Loss
        # --------------------------------------------------
        total_loss = (
            self.similarity_weight * similarity_loss +
            self.contrastive_weight * contrastive_loss
        )

        return {
            "total_loss": total_loss,
            "similarity_loss": similarity_loss,
            "contrastive_loss": contrastive_loss
        }