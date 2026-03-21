"""
train_ocular_v2.py  —  Balanced sensitivity/specificity training run
---------------------------------------------------------------------
Key differences vs train_ocular.py:
  • pos_weight = 0.35  (down-weights CKD to reduce over-prediction of the majority class)
  • Manual label smoothing (BCEWithLogitsLoss has no built-in label_smoothing arg)
  • Separated augmentation transform (train only) from base transform (val only)
  • PIL import at module level so DataLoader workers can find it

Run from the project root:
    python scripts/train_ocular_v2.py
"""

import os
import sys

# Allow sibling-imports (ocular_model.py lives in the same scripts/ dir)
sys.path.insert(0, os.path.dirname(__file__))

from PIL import Image                        # ← FIX 1: must be at module level so
                                             #   DataLoader worker processes can import it

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as TF
from torch.optim import NAdam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data.sampler import WeightedRandomSampler

from ocular_model import BilateralCKDModel, GrahamTransform

cudnn.benchmark = True


# ── Custom loss with Focal Loss ───────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Focal Loss targeting True Class Imbalance & Hard Examples.
    Replaces SmoothedBCEWithLogitsLoss to force the model to focus on the harder minority class.
    """
    def __init__(self, pos_weight=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = pos_weight if pos_weight is not None else torch.tensor([0.5])
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        
        # alpha weighting for targets vs non-targets
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        
        focal_loss = alpha_t * (1 - pt)**self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# ── Dataset ───────────────────────────────────────────────────────────────────
class BilateralDataset(Dataset):
    """
    Loads a single retinal image and produces a pseudo-bilateral pair:
      • left_img  — original (optionally augmented)
      • right_img — horizontal mirror of the processed left image
    """
    def __init__(self, data_df, img_dir, transform=None):
        self.data_df = data_df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        img_id   = self.data_df.iloc[idx]['ID']
        img_path = os.path.join(self.img_dir, f"{int(img_id)}.png")

        image = Image.open(img_path).convert('RGB')   # PIL is now importable in workers

        if self.transform:
            image = self.transform(image)

        label = torch.tensor(
            self.data_df.iloc[idx]['Diagnosis'], dtype=torch.float32
        ).unsqueeze(0)

        image_left  = image
        image_right = TF.hflip(image)   # pseudo right-eye via horizontal flip

        return image_left, image_right, label


# ── Training ──────────────────────────────────────────────────────────────────
def train_fold(fold_idx, train_loader, val_loader, hparams, device):
    print(f"\n--- Starting Fold {fold_idx + 1} ---")
    model = BilateralCKDModel().to(device)

    criterion = FocalLoss(
        pos_weight=torch.tensor([0.5]).to(device), # Balanced 50/50
        gamma=2.0
    )

    optimizer = NAdam(model.parameters(), lr=hparams['lr'],
                      weight_decay=hparams['weight_decay'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    scaler    = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    best_val_loss = float('inf')

    for epoch in range(hparams['epochs']):
        # ── Train ──
        model.train()
        train_loss = 0.0
        for left, right, labels in train_loader:
            left, right, labels = left.to(device), right.to(device), labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                outputs = model(left, right)
                loss    = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        # ── Validate ──
        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for left, right, labels in val_loader:
                left, right, labels = left.to(device), right.to(device), labels.to(device)
                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    outputs = model(left, right)
                    loss    = criterion(outputs, labels)

                val_loss += loss.item()
                preds     = (torch.sigmoid(outputs) > 0.5).float()
                correct  += (preds == labels).sum().item()
                total    += labels.size(0)

        avg_train = train_loss / len(train_loader)
        avg_val   = val_loss   / len(val_loader)
        acc       = correct / total

        print(f"Epoch {epoch+1:02d}: Train Loss {avg_train:.4f} | "
              f"Val Loss {avg_val:.4f} | Acc {acc:.4f}")

        scheduler.step(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            os.makedirs("models", exist_ok=True)
            save_path = f"models/fold_{fold_idx + 1}.pth"
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Saved best weights → {save_path}")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    hparams = {
        'lr'              : 1e-4,   # Bumped from 1e-5 to shatter the trivial classifier plateau
        'pos_weight'      : 0.35,   # < 1 → reduces over-prediction of majority CKD class
        'label_smoothing' : 0.1,    # prevents overconfidence
        'weight_decay'    : 1e-4,   # L2 regularisation
        'epochs'          : 15,
        'batch_size'      : 16,     # Bumped up, but capped at 16 (Bilateral EfficientNet-B3 demands high VRAM)
        'img_dir'         : 'dataset/images',
        'num_workers'     : 0       # MUST BE 0 ON WINDOWS to prevent [WinError 5] Access is denied
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # ── Load data ──
    # Using the RFMiD Training Labels for general disease risk
    df = pd.read_csv("dataset/RFMiD_Training_Labels.csv")
    
    # Use the explicit Disease_Risk column as the target
    df['Diagnosis'] = df['Disease_Risk']
    
    print(f"Dataset stats: {len(df)} total images.")
    print("Class Distribution:\n", df['Diagnosis'].value_counts())

    skf    = StratifiedGroupKFold(n_splits=5)
    groups = df['ID'] 
    X, y   = df.index, df['Diagnosis']

    # ── Transforms ──
    # FIX 3: Separate transforms — augmentation only on training data.
    # Validation MUST be deterministic (no random flips/rotations/jitter).
    train_transform = transforms.Compose([
        GrahamTransform(sigma=10),
        transforms.Resize((300, 300)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        GrahamTransform(sigma=10),
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # ── Cross-validation loop ──
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y, groups=groups)):
        train_df = df.iloc[train_idx]
        val_df   = df.iloc[val_idx]

        train_ds = BilateralDataset(train_df, hparams['img_dir'], transform=train_transform)
        val_ds   = BilateralDataset(val_df,   hparams['img_dir'], transform=val_transform)

        # 🚀 CUSTOM BALANCED SAMPLER LOGIC
        # Force every training batch to have a 50/50 mix of Healthy vs CKD
        train_labels = train_df['Diagnosis'].values.astype(int)
        class_sample_count = np.array([len(np.where(train_labels == t)[0]) for t in np.unique(train_labels)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in train_labels])
        
        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(samples_weight),
            num_samples=len(samples_weight),
            replacement=True
        )

        train_loader = DataLoader(train_ds, batch_size=hparams['batch_size'],
                                  sampler=sampler, # Sampler handles balancing
                                  num_workers=0, pin_memory=True)
                                  
        val_loader   = DataLoader(val_ds,   batch_size=hparams['batch_size'],
                                  shuffle=False, num_workers=0, pin_memory=True)

        train_fold(fold_idx, train_loader, val_loader, hparams, device)

    print("\n✓ All 5 folds complete.")


if __name__ == "__main__":
    main()
