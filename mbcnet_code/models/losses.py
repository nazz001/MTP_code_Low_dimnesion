"""
Pluggable loss functions for metric learning
Add new loss functions here and register in get_loss_function()
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def pairwise_distance(embeddings):
    """Compute pairwise cosine distance (1 - cosine similarity)"""
    return 1 - torch.matmul(embeddings, embeddings.T)


class TripletLoss(nn.Module):
    """
    Triplet Loss with multiple mining strategies

    Args:
        margin: Margin for triplet loss
        mode: Mining strategy ("batch_hard", "batch_all", "semi_hard")
    """
    def __init__(self, margin=0.3, mode="batch_hard"):
        super().__init__()
        self.margin = margin
        self.mode = mode

    def forward(self, embeddings, labels):
        if self.mode == "batch_hard":
            return self._batch_hard(embeddings, labels)
        elif self.mode == "batch_all":
            return self._batch_all(embeddings, labels)
        elif self.mode == "semi_hard":
            return self._semi_hard(embeddings, labels)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def _batch_hard(self, embeddings, labels):
        # 
        """Batch hard triplet loss"""
        dist_matrix = pairwise_distance(embeddings)
        labels = labels.unsqueeze(1)

        # Find hardest positive
        pos_mask = (labels == labels.T).float()
        pos_dist = dist_matrix.clone()
        pos_dist[pos_mask == 0] = -1
        hardest_positive = pos_dist.max(dim=1)[0].clamp(min=0)

        # Find hardest negative
        neg_mask = (labels != labels.T).float()
        neg_dist = dist_matrix.clone()
        neg_dist[neg_mask == 0] = 1e6
        hardest_negative = neg_dist.min(dim=1)[0]

        loss = torch.relu(self.margin + hardest_positive - hardest_negative)
        return loss.mean()

    def _batch_all(self, embeddings, labels):
        """Batch all triplet loss"""
        dist_matrix = pairwise_distance(embeddings)
        batch_size = embeddings.size(0)

        # Create masks
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        i_not_equal_j = ~torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
        i_not_equal_k = i_not_equal_j

        # Valid triplets
        valid_positives = labels_equal & i_not_equal_j
        valid_negatives = ~labels_equal & i_not_equal_k

        # Compute loss for all valid triplets
        triplet_loss = []
        for i in range(batch_size):
            pos_dists = dist_matrix[i][valid_positives[i]]
            neg_dists = dist_matrix[i][valid_negatives[i]]

            if len(pos_dists) > 0 and len(neg_dists) > 0:
                pos_dists = pos_dists.unsqueeze(1)
                neg_dists = neg_dists.unsqueeze(0)
                loss = torch.relu(self.margin + pos_dists - neg_dists)
                triplet_loss.append(loss.mean())

        if len(triplet_loss) > 0:
            return torch.stack(triplet_loss).mean()
        return torch.tensor(0.0, device=embeddings.device)

    def _semi_hard(self, embeddings, labels):
        """Semi-hard triplet loss"""
        dist_matrix = pairwise_distance(embeddings)
        labels = labels.unsqueeze(1)

        # Positive mask
        pos_mask = (labels == labels.T).float()
        pos_dist = dist_matrix * pos_mask

        # Negative mask
        neg_mask = (labels != labels.T).float()

        # Find semi-hard negatives
        losses = []
        for i in range(len(embeddings)):
            pos_dists = dist_matrix[i][pos_mask[i] == 1]
            neg_dists = dist_matrix[i][neg_mask[i] == 1]

            if len(pos_dists) > 0 and len(neg_dists) > 0:
                hardest_pos = pos_dists.max()

                # Semi-hard: negatives where pos < neg < pos + margin
                semi_hard_neg = neg_dists[(neg_dists > hardest_pos) & 
                                         (neg_dists < hardest_pos + self.margin)]

                if len(semi_hard_neg) > 0:
                    loss = torch.relu(self.margin + hardest_pos - semi_hard_neg.min())
                    losses.append(loss)
                else:
                    # Fallback to hardest negative
                    loss = torch.relu(self.margin + hardest_pos - neg_dists.min())
                    losses.append(loss)

        if len(losses) > 0:
            return torch.stack(losses).mean()
        return torch.tensor(0.0, device=embeddings.device)


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss for pairs

    Args:
        margin: Margin for negative pairs
    """
    def __init__(self, margin=0.5):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: [batch_size, embed_dim]
            labels: [batch_size]
        """
        batch_size = embeddings.size(0)

        # Compute pairwise distances
        dist_matrix = pairwise_distance(embeddings)

        # Create pair labels (1 for same class, 0 for different)
        labels = labels.unsqueeze(1)
        pair_labels = (labels == labels.T).float()

        # Loss for positive pairs
        pos_loss = pair_labels * dist_matrix.pow(2)

        # Loss for negative pairs
        neg_loss = (1 - pair_labels) * torch.relu(self.margin - dist_matrix).pow(2)

        # Exclude diagonal
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)

        loss = (pos_loss + neg_loss)[mask].mean()
        return loss


class ArcFaceLoss(nn.Module):
    """
    ArcFace Loss (Angular Margin Loss)

    Args:
        in_features: Feature dimension
        out_features: Number of classes
        scale: Scale parameter (s)
        margin: Angular margin (m)
    """
    def __init__(self, in_features, out_features, scale=30.0, margin=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin

        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        """
        Args:
            embeddings: [batch_size, in_features]
            labels: [batch_size]
        """
        # Normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)

        # Cosine similarity
        cosine = F.linear(embeddings, weight)

        # Get target logits
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1)

        # Add angular margin
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logits = torch.cos(theta + self.margin)

        # Apply scale
        output = cosine * (1 - one_hot) + target_logits * one_hot
        output *= self.scale

        return F.cross_entropy(output, labels)


def get_loss_function(loss_type, **kwargs):
    """
    Factory function to get loss function

    Args:
        loss_type: Type of loss ("triplet", "contrastive", "arcface")
        **kwargs: Additional arguments for loss function

    Returns:
        Loss function instance
    """
    loss_registry = {
        "triplet": TripletLoss,
        "contrastive": ContrastiveLoss,
        "arcface": ArcFaceLoss,
    }

    if loss_type not in loss_registry:
        raise ValueError(f"Unknown loss type: {loss_type}. "
                        f"Available: {list(loss_registry.keys())}")

    loss_class = loss_registry[loss_type]
    return loss_class(**kwargs)