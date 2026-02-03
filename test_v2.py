import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve

# =====================================================
# DEVICE
# =====================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n🔍 Testing on device: {device}\n")

# =====================================================
# TRANSFORM (MUST MATCH test_tf in training)
# =====================================================
test_tf = transforms.Compose([
    transforms.Resize((64, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# =====================================================
# MODEL DEFINITIONS (EXACT COPY FROM TRAIN V2)
# =====================================================
class ResidualChannelAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c // 4, c, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x + x * self.att(x)


class MultiLayerBranch(nn.Module):
    def __init__(self, k, d):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, k, padding=d, dilation=d, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResidualChannelAttention(32)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResidualChannelAttention(32)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            ResidualChannelAttention(32)
        )

        self.down1 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.down1(x)
        x = self.down2(x)
        return x


class MBLNet(nn.Module):
    def __init__(self, emb_dim=256):
        super().__init__()

        self.b1 = MultiLayerBranch(3, 1)
        self.b2 = MultiLayerBranch(5, 1)
        self.b3 = MultiLayerBranch(3, 2)

        self.fusion = nn.Sequential(
            nn.Conv2d(96, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, emb_dim)
        )

    def forward(self, x):
        f = torch.cat([self.b1(x), self.b2(x), self.b3(x)], dim=1)
        f = self.fusion(f)
        f = self.pool(f).flatten(1)
        return F.normalize(self.fc(f), dim=1)

# =====================================================
# LOAD MODEL
# =====================================================
model = MBLNet().to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()
print("✅ Loaded best_model.pth")

# =====================================================
# EER COMPUTATION
# =====================================================
@torch.no_grad()
def compute_eer(csv_file):
    df = pd.read_csv(csv_file)

    scores, labels = [], []

    for _, r in df.iterrows():
        img1 = test_tf(Image.open(r.iloc[0]).convert("L")).unsqueeze(0).to(device)
        img2 = test_tf(Image.open(r.iloc[2]).convert("L")).unsqueeze(0).to(device)

        f1 = model(img1)
        f2 = model(img2)

        scores.append(F.cosine_similarity(f1, f2).item())
        labels.append(int(r.iloc[4]))

    scores = np.array(scores)
    labels = np.array(labels)

    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer = fpr[np.argmin(np.abs(fpr - fnr))]

    return eer

# =====================================================
# MAIN
# =====================================================
if __name__ == "__main__":
    test_csv = "/home/nishkal/alam/abletion_study_and_feature_extration/abletion/casia_v1_testpair.csv"

    eer = compute_eer(test_csv)
    print(f"\n🎯 Test EER: {eer * 100:.2f}%\n")

# eer 10.
