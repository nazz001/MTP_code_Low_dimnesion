import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve
from tqdm import tqdm
import random

# =====================================================
# REPRODUCIBILITY
# =====================================================
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# =====================================================
# MODEL
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
# DATASET (PAIR CSV)
# =====================================================

class IrisPairDataset(Dataset):
    def __init__(self, csv_file, transform):
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        img1 = self.transform(Image.open(r.iloc[0]).convert("L"))
        img2 = self.transform(Image.open(r.iloc[2]).convert("L"))
        label = torch.tensor(r.iloc[4], dtype=torch.float32)
        return img1, img2, label

# =====================================================
# LOSS
# =====================================================

class CosineContrastiveLoss(nn.Module):
    def __init__(self, margin=0.4):
        super().__init__()
        self.margin = margin

    def forward(self, f1, f2, y):
        cos = F.cosine_similarity(f1, f2)
        pos = (1 - cos) * y
        neg = F.relu(cos - self.margin) * (1 - y)
        return (pos + neg).mean()

# =====================================================
# EER
# =====================================================

@torch.no_grad()
def compute_eer(model, csv_file, transform, device):
    model.eval()
    df = pd.read_csv(csv_file)
    scores, labels = [], []

    for _, r in df.iterrows():
        img1 = transform(Image.open(r.iloc[0]).convert("L")).unsqueeze(0).to(device)
        img2 = transform(Image.open(r.iloc[2]).convert("L")).unsqueeze(0).to(device)
        f1 = model(img1)
        f2 = model(img2)
        scores.append(F.cosine_similarity(f1, f2).item())
        labels.append(int(r.iloc[4]))

    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    eer = fpr[np.argmin(np.abs(fpr - fnr))]
    return eer

# =====================================================
# TRAINING
# =====================================================

def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n🚀 Training on {device}\n")

    # -------- transforms (FIXED ORDER) --------
    # train_tf = transforms.Compose([
    #     transforms.Resize((72, 280)),
    #     transforms.RandomCrop((64, 256)),
    #     transforms.RandomAffine(5, translate=(0.05,0.05), scale=(0.9,1.1)),
    #     transforms.ColorJitter(brightness=0.2, contrast=0.3),
    #     transforms.RandomErasing(p=0.2),
    #     transforms.ToTensor(),                  # ✅ MUST come first
    #     transforms.Normalize([0.5], [0.5])
    # ])

    # test_tf = transforms.Compose([
    #     transforms.Resize((64, 256)),
    #     transforms.ToTensor(),                  # ✅ MUST come first
    #     transforms.Normalize([0.5], [0.5])
    # ])
    train_tf = transforms.Compose([
    transforms.Resize((72, 280)),
    transforms.RandomCrop((64, 256)),
    transforms.RandomAffine(
        degrees=5,
        translate=(0.05, 0.05),
        scale=(0.9, 1.1)
    ),
    transforms.ColorJitter(brightness=0.2, contrast=0.3),

    transforms.ToTensor(),                  # ✅ PIL → Tensor happens here

    transforms.RandomErasing(               # ✅ NOW SAFE
        p=0.2,
        scale=(0.02, 0.1),
        ratio=(0.3, 3.3)
    ),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
    
    test_tf = transforms.Compose([
    transforms.Resize((64, 256)),
    transforms.ToTensor(),                  # ✅ PIL → Tensor
    transforms.Normalize(mean=[0.5], std=[0.5])
])



    # -------- data --------
    train_ds = IrisPairDataset("/home/nishkal/alam/abletion_study_and_feature_extration/abletion/splits/train_pairs.csv", train_tf)
    val_ds   = IrisPairDataset("/home/nishkal/alam/abletion_study_and_feature_extration/abletion/splits/val_pairs.csv", test_tf)

    train_loader = DataLoader(
        train_ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
    )

    # -------- model --------
    model = MBLNet().to(device)
    criterion = CosineContrastiveLoss(margin=0.4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    best_eer = 1.0

    for epoch in range(1, 51):
        model.train()
        total_loss = 0.0

        for x1, x2, y in tqdm(train_loader, desc=f"Epoch {epoch:02d}", ncols=120):
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)

            f1 = model(x1)
            f2 = model(x2)
            loss = criterion(f1, f2, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        eer = compute_eer(model, "/home/nishkal/alam/abletion_study_and_feature_extration/abletion/splits/val_pairs.csv", test_tf, device)

        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {total_loss/len(train_loader):.4f} | "
            f"Val EER: {eer*100:.2f}%"
        )

        if eer < best_eer:
            best_eer = eer
            torch.save(model.state_dict(), "best_model.pth")
            print("✅ Best model saved")

    print(f"\n🏆 Training complete. Best Val EER: {best_eer*100:.2f}%\n")

# =====================================================
if __name__ == "__main__":
    train()
