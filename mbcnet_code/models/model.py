"""
Multi-Branch Learning Network (MBLNet)
Modular architecture with pluggable components
"""
import torch
import torch.nn as nn
from .attention import get_attention_module


def conv_block(in_channels, out_channels, kernel_size, padding=0, dilation=1):
    """Standard convolution block with BatchNorm and ReLU"""
    return nn.Sequential(
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
# conv block takes input and convert into the channel given and then batch norm and relu return


class Branch(nn.Module):
    """Single branch in multi-branch architecture"""

    def __init__(
        self,
        in_channels,
        channel_list,
        kernel_size,
        dilation,
        attention_type="channel",
        attention_kwargs=None
    ):
        super().__init__()

        if attention_kwargs is None:
            attention_kwargs = {}

        branch_layers = []

        for layer_idx, out_channels in enumerate(channel_list):

            input_channels = (
                in_channels if layer_idx == 0 else channel_list[layer_idx - 1]
            )
            # first layer read the image later leayers read the prevouis stage

            # Convolution block
            padding = (kernel_size // 2) * dilation
            branch_layers.append(
                conv_block(
                    input_channels,
                    out_channels,
                    kernel_size,
                    padding,
                    dilation
                )
            )

            # Attention module
            attention = get_attention_module(
                attention_type,
                out_channels,
                **attention_kwargs
            )
            branch_layers.append(attention)

            # Pooling (except last layer)
            if layer_idx < len(channel_list) - 1:
                branch_layers.append(nn.AvgPool2d(kernel_size=2))

        # Global pooling
        branch_layers.append(nn.AdaptiveAvgPool2d((1, 1)))

        self.branch = nn.Sequential(*branch_layers)

    def forward(self, x):
        return self.branch(x)


class MBLNet(nn.Module):
    """
    Multi-Branch Learning Network

    Args:
        in_channels: Input channels (1 for grayscale)
        channel_list: List of channel dimensions for each stage
        branch_configs: List of (kernel_size, dilation) for each branch
        embed_dim: Embedding dimension
        feature_dim: Final feature dimension
        attention_type: Type of attention mechanism
        attention_kwargs: Additional arguments for attention
    """

    def __init__(
        self,
        in_channels=1,
        channel_list=[16, 24, 32, 48, 64],
        branch_configs=[(3, 1), (5, 1), (3, 2)],
        embed_dim=256,
        feature_dim=128,
        attention_type="channel",
        attention_kwargs=None
    ):
        # bracnh configuration means first kernal size and second the dilartion
        # embediding dim main dimnesion 
        # fearure dimension measn dimnesion used during training
        super().__init__()

        if attention_kwargs is None:
            attention_kwargs = {}

        # Build branches
        self.branches = nn.ModuleList()
        for kernel_size, dilation in branch_configs:
            self.branches.append(
                Branch(
                    in_channels=in_channels,
                    channel_list=channel_list,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    attention_type=attention_type,
                    attention_kwargs=attention_kwargs
                )
            )

        # Fusion and embedding
        fusion_dim = channel_list[-1] * len(branch_configs)

        self.embedding_head = nn.Sequential(
            nn.Linear(fusion_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )
        

        # Projection head for training
        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),

            nn.Linear(embed_dim, feature_dim)
        )

        self.embed_dim = embed_dim
        self.feature_dim = feature_dim

    def forward(self, x, return_embedding=False):

        # Extract features from all branches
        branch_outputs = []
        for branch in self.branches:
            features = branch(x)
            features = features.view(x.size(0), -1)
            branch_outputs.append(features)

        # Concatenate and embed
        fused_features = torch.cat(branch_outputs, dim=1)

        embedding = self.embedding_head(fused_features)
        embedding = embedding / (embedding.norm(dim=1, keepdim=True) + 1e-8)

        if return_embedding:
            return embedding

        # Project to feature space
        projected_features = self.projection_head(embedding)
        projected_features = (
            projected_features
            / (projected_features.norm(dim=1, keepdim=True) + 1e-8)
        )

        return projected_features


def build_model(config):
    """
    Build model from configuration

    Args:
        config: Configuration module or dict

    Returns:
        Model instance
    """

    if hasattr(config, "__dict__"):
        cfg = config
    else:
        cfg = type("Config", (), config)()

    attention_kwargs = {
        "kernel_size": getattr(cfg, "ATTENTION_KERNEL", 5)
    }

    model = MBLNet(
        in_channels=getattr(cfg, "IMG_CHANNELS", 1),
        channel_list=getattr(cfg, "BACKBONE_CHANNELS", [16, 24, 32, 48, 64]),
        branch_configs=getattr(
            cfg,
            "BRANCH_CONFIGS",
            [(3, 1), (5, 1), (3, 2)]
        ),
        embed_dim=getattr(cfg, "EMBED_DIM", 256),
        feature_dim=getattr(cfg, "FEATURE_DIM", 128),
        attention_type=getattr(cfg, "ATTENTION_TYPE", "channel"),
        attention_kwargs=attention_kwargs
    )

    return model
