"""
Pluggable attention mechanisms
Add new attention modules here and register in get_attention_module()
"""
import torch
import torch.nn as nn


# class ChannelAttn(nn.Module):
#     """Channel Attention Module"""
#     def __init__(self, channels, kernel_size=5):
#         super().__init__()
#         self.conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size//2)
#         self.act = nn.ReLU()
#         self.sig = nn.Sigmoid()

#     def forward(self, x):
#         B, C, _, _ = x.shape
#         # Aggregate spatial information
#         avg_pool = x.mean((2, 3))
#         max_pool = x.view(B, C, -1).max(2)[0]
#         f = avg_pool + max_pool

#         # Generate attention weights
#         w = self.sig(self.act(self.conv(f.unsqueeze(1))))
#         return x * w.view(B, C, 1, 1)


class ChannelAttn(nn.Module):
    """Channel Attention Module (simple & readable)"""

    def __init__(self, channels, kernel_size=5):
        super().__init__()

        # Learn channel-wise relationships
        self.channel_conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        x: (B, C, H, W)
        """

        B, C, H, W = x.shape

        # 1. Global Average Pooling → (B, C)
        avg_pool = x.mean(dim=(2, 3))

        # 2. Global Max Pooling → (B, C)
        max_pool = x.view(B, C, H * W).max(dim=2)[0]

        # 3. Combine channel descriptors
        channel_descriptor = avg_pool + max_pool   # (B, C)

        # 4. Generate channel attention weights
        channel_descriptor = channel_descriptor.unsqueeze(1)  # (B, 1, C)
        attention = self.channel_conv(channel_descriptor)      # (B, 1, C)
        # attention = self.relu(attention)
        attention = self.sigmoid(attention)

        # 5. Apply channel attention
        attention = attention.view(B, C, 1, 1)
        return x * attention





import torch
import torch.nn as nn


# --------------------------------------------------
# Spatial Attention
# --------------------------------------------------
class SpatialAttention(nn.Module):
    """
    Spatial Attention Module
    Learns WHERE to focus in the feature map
    """

    def __init__(self, kernel_size=7):
        super().__init__()

        # Takes 2-channel input (avg + max) and outputs 1 attention map
        self.spatial_conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        x: (B, C, H, W)
        """

        # 1. Channel-wise average pooling → (B, 1, H, W)
        avg_map = x.mean(dim=1, keepdim=True)

        # 2. Channel-wise max pooling → (B, 1, H, W)
        max_map = x.max(dim=1, keepdim=True)[0]

        # 3. Concatenate along channel dimension → (B, 2, H, W)
        spatial_descriptor = torch.cat([avg_map, max_map], dim=1)

        # 4. Generate spatial attention map → (B, 1, H, W)
        spatial_attention = self.sigmoid(
            self.spatial_conv(spatial_descriptor)
        )

        # 5. Apply spatial attention
        return x * spatial_attention


# --------------------------------------------------
# CBAM = Channel Attention + Spatial Attention
# --------------------------------------------------
class CBAM(nn.Module):
    """
    Convolutional Block Attention Module
    Applies Channel Attention first, then Spatial Attention
    """

    def __init__(self, channels, kernel_size=7):
        super().__init__()

        self.channel_attention = ChannelAttn(channels)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# --------------------------------------------------
# Identity (No Attention)
# --------------------------------------------------
class IdentityAttention(nn.Module):
    """
    Identity attention (does nothing)
    Useful for ablation studies
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


# --------------------------------------------------
# Attention Factory Function
# --------------------------------------------------
def build_attention_module(attention_type, channels, **kwargs):
    """
    Factory function to build attention modules

    Args:
        attention_type (str): "channel", "spatial", "cbam", or "none"
        channels (int): Number of feature channels
        **kwargs: Optional arguments (e.g., kernel_size)

    Returns:
        nn.Module: Attention module
    """

    attention_registry = {
        "channel": ChannelAttn,
        "spatial": SpatialAttention,
        "cbam": CBAM,
        "none": IdentityAttention,
    }

    if attention_type not in attention_registry:
        raise ValueError(
            f"Unknown attention type: {attention_type}. "
            f"Available types: {list(attention_registry.keys())}"
        )

    kernel_size = kwargs.get("kernel_size", 7)

    if attention_type == "channel":
        return ChannelAttn(channels, kernel_size)

    elif attention_type == "spatial":
        return SpatialAttention(kernel_size)

    elif attention_type == "cbam":
        return CBAM(channels, kernel_size)

    else:  # "none"
        return IdentityAttention()




# class SpatialAttn(nn.Module):
#     """Spatial Attention Module"""
#     def __init__(self, kernel_size=7):
#         super().__init__()
#         self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
#         self.sig = nn.Sigmoid()

#     def forward(self, x):
#         # Aggregate channel information
#         avg_pool = x.mean(dim=1, keepdim=True)
#         max_pool = x.max(dim=1, keepdim=True)[0]
#         concat = torch.cat([avg_pool, max_pool], dim=1)

#         # Generate attention map
#         attn = self.sig(self.conv(concat))
#         return x * attn


# class CBAM(nn.Module):
#     """Convolutional Block Attention Module (Channel + Spatial)"""
#     def __init__(self, channels, kernel_size=7):
#         super().__init__()
#         self.channel_attn = ChannelAttn(channels)
#         self.spatial_attn = SpatialAttn(kernel_size)

#     def forward(self, x):
#         x = self.channel_attn(x)
#         x = self.spatial_attn(x)
#         return x


# class NoAttention(nn.Module):
#     """Identity module (no attention)"""
#     def __init__(self, *args, **kwargs):
#         super().__init__()

#     def forward(self, x):
#         return x


# def get_attention_module(attn_type, channels, **kwargs):
#     """
#     Factory function to get attention module

#     Args:
#         attn_type: Type of attention ("channel", "spatial", "cbam", "none")
#         channels: Number of channels
#         **kwargs: Additional arguments for attention module

#     Returns:
#         Attention module instance
#     """
#     attn_registry = {
#         "channel": ChannelAttn,
#         "spatial": SpatialAttn,
#         "cbam": CBAM,
#         "none": NoAttention,
#     }

#     if attn_type not in attn_registry:
#         raise ValueError(f"Unknown attention type: {attn_type}. "
#                         f"Available: {list(attn_registry.keys())}")

#     attn_class = attn_registry[attn_type]

#     # Build appropriate arguments
#     if attn_type == "channel":
#         return attn_class(channels, kwargs.get("kernel_size", 5))
#     elif attn_type == "spatial":
#         return attn_class(kwargs.get("kernel_size", 7))
#     elif attn_type == "cbam":
#         return attn_class(channels, kwargs.get("kernel_size", 7))
#     else:  # none
#         return attn_class()