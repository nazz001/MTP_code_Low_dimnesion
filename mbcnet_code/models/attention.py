"""
Pluggable attention mechanisms
Add new attention modules here and register in get_attention_module()
"""
import torch
import torch.nn as nn


class ChannelAttn(nn.Module):
    """Channel Attention Module"""
    def __init__(self, channels, kernel_size=5):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size//2)
        self.act = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        B, C, _, _ = x.shape
        # Aggregate spatial information
        avg_pool = x.mean((2, 3))
        max_pool = x.view(B, C, -1).max(2)[0]
        f = avg_pool + max_pool

        # Generate attention weights
        w = self.sig(self.act(self.conv(f.unsqueeze(1))))
        return x * w.view(B, C, 1, 1)


class SpatialAttn(nn.Module):
    """Spatial Attention Module"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # Aggregate channel information
        avg_pool = x.mean(dim=1, keepdim=True)
        max_pool = x.max(dim=1, keepdim=True)[0]
        concat = torch.cat([avg_pool, max_pool], dim=1)

        # Generate attention map
        attn = self.sig(self.conv(concat))
        return x * attn


class CBAM(nn.Module):
    """Convolutional Block Attention Module (Channel + Spatial)"""
    def __init__(self, channels, kernel_size=7):
        super().__init__()
        self.channel_attn = ChannelAttn(channels)
        self.spatial_attn = SpatialAttn(kernel_size)

    def forward(self, x):
        x = self.channel_attn(x)
        x = self.spatial_attn(x)
        return x


class NoAttention(nn.Module):
    """Identity module (no attention)"""
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


def get_attention_module(attn_type, channels, **kwargs):
    """
    Factory function to get attention module

    Args:
        attn_type: Type of attention ("channel", "spatial", "cbam", "none")
        channels: Number of channels
        **kwargs: Additional arguments for attention module

    Returns:
        Attention module instance
    """
    attn_registry = {
        "channel": ChannelAttn,
        "spatial": SpatialAttn,
        "cbam": CBAM,
        "none": NoAttention,
    }

    if attn_type not in attn_registry:
        raise ValueError(f"Unknown attention type: {attn_type}. "
                        f"Available: {list(attn_registry.keys())}")

    attn_class = attn_registry[attn_type]

    # Build appropriate arguments
    if attn_type == "channel":
        return attn_class(channels, kwargs.get("kernel_size", 5))
    elif attn_type == "spatial":
        return attn_class(kwargs.get("kernel_size", 7))
    elif attn_type == "cbam":
        return attn_class(channels, kwargs.get("kernel_size", 7))
    else:  # none
        return attn_class()