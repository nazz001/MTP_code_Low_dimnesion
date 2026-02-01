# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # ============================================================
# # ATTENTION MODULES
# # ============================================================

# class ChannelAttn(nn.Module):
#     def __init__(self, channels, kernel_size=5):
#         super().__init__()
#         self.conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size // 2)
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()


#     def forward(self, x):
#         B, C, _, _ = x.shape
#         # shape bracnh and chanel and height and weigh
#         avg_pool = x.mean(dim=(2, 3))
#         # dim tell takes means over second and third which are height and wideth
#         # output bracnh and channel Each channel is reduced to one number
#         max_pool = x.view(B, C, -1).max(dim=2)[0]
#         # here orignal shape x.shape = (B, C, H, W) convert into  (B, C, H*W)
#         # now .max(mid==2)  find max in each change  and then here [0] keep only max discard indieced





#         pooled = avg_pool + max_pool
#         weights = self.sigmoid(self.relu(self.conv(pooled.unsqueeze(1))))
#         return x * weights.view(B, C, 1, 1)


# class SpatialAttn(nn.Module):
#     def __init__(self, kernel_size=7):
#         super().__init__()
#         self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, x):
#         avg_pool = x.mean(dim=1, keepdim=True)
#         max_pool = x.max(dim=1, keepdim=True)[0]
#         concat = torch.cat([avg_pool, max_pool], dim=1)
#         return x * self.sigmoid(self.conv(concat))


# class CBAM(nn.Module):
#     def __init__(self, channels, kernel_size=7):
#         super().__init__()
#         self.channel = ChannelAttn(channels)
#         self.spatial = SpatialAttn(kernel_size)

#     def forward(self, x):
#         x = self.channel(x)
#         return self.spatial(x)


# # ============================================================
# # BASIC CONV BLOCK
# # ============================================================

# def conv_block(in_channels, out_channels, kernel_size, padding, dilation):
#     return nn.Sequential(
#         nn.Conv2d(
#             in_channels,
#             out_channels,
#             kernel_size,
#             padding=padding,
#             dilation=dilation
#         ),
#         nn.BatchNorm2d(out_channels),
#         nn.ReLU(inplace=True)
#     )


# # ============================================================
# # SINGLE BRANCH (NO LOOP INSIDE)
# # ============================================================

# class Branch(nn.Module):
#     def __init__(
#         self,
#         in_channels,
#         c1, c2, c3, c4, c5,
#         kernel_size,
#         dilation,
#         attention_type="channel",
#         attention_kernel=5
#     ):
#         super().__init__()

#         padding = (kernel_size // 2) * dilation

#         # Stage 1
#         self.block1 = conv_block(in_channels, c1, kernel_size, padding, dilation)
#         self.attn1 = ChannelAttn(c1) if attention_type == "channel" else CBAM(c1)
#         self.pool1 = nn.AvgPool2d(2)

#         # Stage 2
#         self.block2 = conv_block(c1, c2, kernel_size, padding, dilation)
#         self.attn2 = ChannelAttn(c2) if attention_type == "channel" else CBAM(c2)
#         self.pool2 = nn.AvgPool2d(2)

#         # Stage 3
#         self.block3 = conv_block(c2, c3, kernel_size, padding, dilation)
#         self.attn3 = ChannelAttn(c3) if attention_type == "channel" else CBAM(c3)
#         self.pool3 = nn.AvgPool2d(2)

#         # Stage 4
#         self.block4 = conv_block(c3, c4, kernel_size, padding, dilation)
#         self.attn4 = ChannelAttn(c4) if attention_type == "channel" else CBAM(c4)
#         self.pool4 = nn.AvgPool2d(2)

#         # Stage 5
#         self.block5 = conv_block(c4, c5, kernel_size, padding, dilation)
#         self.attn5 = ChannelAttn(c5) if attention_type == "channel" else CBAM(c5)

#         self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

#     def forward(self, x):
#         x = self.pool1(self.attn1(self.block1(x)))
#         x = self.pool2(self.attn2(self.block2(x)))
#         x = self.pool3(self.attn3(self.block3(x)))
#         x = self.pool4(self.attn4(self.block4(x)))
#         x = self.attn5(self.block5(x))
#         return self.global_pool(x)


# # ============================================================
# # IRIS MBLNET BACKBONE (EXPLICIT BRANCHES)
# # ============================================================

# class MBLNet(nn.Module):
#     def __init__(
#         self,
#         in_channels=1,
#         channel_list=(16, 24, 32, 48, 64),
#         embed_dim=256,
#         feature_dim=256,
#         # here as of now i am taking 128 dimension
#         attention_type="channel"
#     ):
#         super().__init__()

#         c1, c2, c3, c4, c5 = channel_list

#         # -------- Explicit Branches --------
#         self.branch_1 = Branch(
#             in_channels, c1, c2, c3, c4, c5,
#             kernel_size=3, dilation=1,
#             attention_type=attention_type
#         )

#         self.branch_2 = Branch(
#             in_channels, c1, c2, c3, c4, c5,
#             kernel_size=5, dilation=1,
#             attention_type=attention_type
#         )

#         self.branch_3 = Branch(
#             in_channels, c1, c2, c3, c4, c5,
#             kernel_size=3, dilation=2,
#             attention_type=attention_type
#         )

#         fusion_dim = c5 * 3

#         # Embedding head
#         self.embedding_head = nn.Sequential(
#             nn.Linear(fusion_dim, embed_dim),
#             nn.BatchNorm1d(embed_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(embed_dim, embed_dim)
#         )

#         # Projection head
#         self.projection_head = nn.Sequential(
#             nn.Linear(embed_dim, embed_dim),
#             nn.BatchNorm1d(embed_dim),
#             nn.ReLU(inplace=True),
#             nn.Linear(embed_dim, feature_dim)
#         )

#         self.feature_dim = feature_dim

#     def forward(self, x, return_embedding=False):
#         f1 = self.branch_1(x).view(x.size(0), -1)
#         f2 = self.branch_2(x).view(x.size(0), -1)
#         f3 = self.branch_3(x).view(x.size(0), -1)

#         fused = torch.cat([f1, f2, f3], dim=1)

#         embedding = self.embedding_head(fused)
#         embedding = F.normalize(embedding, dim=1)

#         # if return_embedding:
#         return embedding

#         # features = self.projection_head(embedding)
#         # return F.normalize(features, dim=1)


# # ============================================================
# # FEATURE EXTRACTOR ADAPTER
# # ============================================================

# class IrisMBLNetExtractor(nn.Module):
#     def __init__(self, backbone):
#         super().__init__()
#         self.backbone = backbone

#     def extract_features(self, x):
#         return self.backbone(x, return_embedding=False)


# # ============================================================
# # FEATURE ENHANCER (PLUGIN POINT)
# # ============================================================

# class IdentityFeatureEnhancer(nn.Module):
#     def forward(self, features, identity=None, pair_label=None):
#         return features


# # ============================================================
# # SIAMESE HEAD
# # ============================================================

# class SiameseHead(nn.Module):
#     def __init__(self, feature_dim):
#         super().__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(feature_dim, 128),
#             nn.ReLU(inplace=True),
#             nn.Linear(128, 1)
#         )

#     def forward(self, f1, f2):
#         return self.fc(torch.abs(f1 - f2))


# # ============================================================
# # FINAL IRIS PAIR MODEL
# # ============================================================

# class IrisSiameseVerificationModel(nn.Module):
#     def __init__(self, backbone, feature_enhancer=None):
#         super().__init__()

#         self.extractor = IrisMBLNetExtractor(backbone)
#         # self.enhancer = feature_enhancer or IdentityFeatureEnhancer()
#         self.siamese_head = SiameseHead(backbone.feature_dim)

#     def forward(self, img1, img2):
#         f1 = self.extractor.extract_features(img1)
#         f2 = self.extractor.extract_features(img2)

#         similarity_score = self.siamese_head(f1, f2)

#         return {
#             "similarity_score": similarity_score,
#             "features": (f1, f2)
#         }



import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================
# ATTENTION MODULES
# ============================================================

class ChannelAttn(nn.Module):
    def __init__(self, channels, kernel_size=5):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size // 2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, C, _, _ = x.shape
        # Average pooling over spatial dimensions
        avg_pool = x.mean(dim=(2, 3))
        # Max pooling over spatial dimensions
        max_pool = x.view(B, C, -1).max(dim=2)[0]

        pooled = avg_pool + max_pool
        weights = self.sigmoid(self.relu(self.conv(pooled.unsqueeze(1))))
        return x * weights.view(B, C, 1, 1)


class SpatialAttn(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = x.mean(dim=1, keepdim=True)
        max_pool = x.max(dim=1, keepdim=True)[0]
        concat = torch.cat([avg_pool, max_pool], dim=1)
        return x * self.sigmoid(self.conv(concat))


class CBAM(nn.Module):
    def __init__(self, channels, kernel_size=7):
        super().__init__()
        self.channel = ChannelAttn(channels)
        self.spatial = SpatialAttn(kernel_size)

    def forward(self, x):
        x = self.channel(x)
        return self.spatial(x)


# ============================================================
# BASIC CONV BLOCK
# ============================================================

def conv_block(in_channels, out_channels, kernel_size, padding, dilation):
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


# ============================================================
# SINGLE BRANCH (NO LOOP INSIDE)
# ============================================================

class Branch(nn.Module):
    def __init__(
        self,
        in_channels,
        c1, c2, c3, c4, c5,
        kernel_size,
        dilation,
        attention_type="channel",
        attention_kernel=5
    ):
        super().__init__()

        padding = (kernel_size // 2) * dilation

        # Stage 1
        self.block1 = conv_block(in_channels, c1, kernel_size, padding, dilation)
        self.attn1 = ChannelAttn(c1) if attention_type == "channel" else CBAM(c1)
        self.pool1 = nn.AvgPool2d(2)

        # Stage 2
        self.block2 = conv_block(c1, c2, kernel_size, padding, dilation)
        self.attn2 = ChannelAttn(c2) if attention_type == "channel" else CBAM(c2)
        self.pool2 = nn.AvgPool2d(2)

        # Stage 3
        self.block3 = conv_block(c2, c3, kernel_size, padding, dilation)
        self.attn3 = ChannelAttn(c3) if attention_type == "channel" else CBAM(c3)
        self.pool3 = nn.AvgPool2d(2)

        # Stage 4
        self.block4 = conv_block(c3, c4, kernel_size, padding, dilation)
        self.attn4 = ChannelAttn(c4) if attention_type == "channel" else CBAM(c4)
        self.pool4 = nn.AvgPool2d(2)

        # Stage 5
        self.block5 = conv_block(c4, c5, kernel_size, padding, dilation)
        self.attn5 = ChannelAttn(c5) if attention_type == "channel" else CBAM(c5)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.pool1(self.attn1(self.block1(x)))
        x = self.pool2(self.attn2(self.block2(x)))
        x = self.pool3(self.attn3(self.block3(x)))
        x = self.pool4(self.attn4(self.block4(x)))
        x = self.attn5(self.block5(x))
        return self.global_pool(x)


# ============================================================
# IRIS MBLNET BACKBONE (EXPLICIT BRANCHES)
# ============================================================

class MBLNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        channel_list=(16, 24, 32, 48, 64),
        embed_dim=256,
        feature_dim=256,
        attention_type="channel"
    ):
        super().__init__()

        c1, c2, c3, c4, c5 = channel_list

        # -------- Explicit Branches --------
        self.branch_1 = Branch(
            in_channels, c1, c2, c3, c4, c5,
            kernel_size=3, dilation=1,
            attention_type=attention_type
        )

        self.branch_2 = Branch(
            in_channels, c1, c2, c3, c4, c5,
            kernel_size=5, dilation=1,
            attention_type=attention_type
        )

        self.branch_3 = Branch(
            in_channels, c1, c2, c3, c4, c5,
            kernel_size=3, dilation=2,
            attention_type=attention_type
        )

        fusion_dim = c5 * 3

        # Embedding head
        self.embedding_head = nn.Sequential(
            nn.Linear(fusion_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim)
        )

        # Projection head
        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, feature_dim)
        )

        self.feature_dim = feature_dim

    def forward(self, x, return_embedding=False):
        f1 = self.branch_1(x).view(x.size(0), -1)
        f2 = self.branch_2(x).view(x.size(0), -1)
        f3 = self.branch_3(x).view(x.size(0), -1)

        fused = torch.cat([f1, f2, f3], dim=1)

        embedding = self.embedding_head(fused)
        embedding = F.normalize(embedding, dim=1)

        return embedding


# ============================================================
# FEATURE EXTRACTOR ADAPTER
# ============================================================

class IrisMBLNetExtractor(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def extract_features(self, x):
        return self.backbone(x, return_embedding=False)


# ============================================================
# FEATURE ENHANCER (PLUGIN POINT)
# ============================================================

class IdentityFeatureEnhancer(nn.Module):
    def forward(self, features, identity=None, pair_label=None):
        return features


# ============================================================
# SIAMESE HEAD
# ============================================================

class SiameseHead(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

    def forward(self, f1, f2):
        return self.fc(torch.abs(f1 - f2))


# ============================================================
# FINAL IRIS PAIR MODEL
# ============================================================

class IrisSiameseVerificationModel(nn.Module):
    def __init__(self, backbone, feature_enhancer=None):
        super().__init__()

        self.extractor = IrisMBLNetExtractor(backbone)
        self.siamese_head = SiameseHead(backbone.feature_dim)

    def forward(self, img1, img2):
        f1 = self.extractor.extract_features(img1)
        f2 = self.extractor.extract_features(img2)

        similarity_score = self.siamese_head(f1, f2)

        return {
            "similarity_score": similarity_score,
            "features": (f1, f2)
        }