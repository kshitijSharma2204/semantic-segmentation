import torch
import matplotlib.pyplot as plt
import os
import numpy as np
from model import UNet
from kitti_dataset import KittiSegmentationDataset
from tqdm import tqdm
from torch.utils.data import random_split

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.devkit_semantics.devkit.helpers.labels import labels


def compute_iou(preds, labels, n_classes):
    ious = []
    preds = preds.view(-1)
    labels = labels.view(-1)
    for cls in range(n_classes):
        pred_inds = preds == cls
        target_inds = labels == cls
        intersection = (pred_inds[target_inds]).long().sum().item()
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
        if union == 0:
            ious.append(float('nan'))  # if no ground truth, do not include in evaluation
        else:
            ious.append(intersection / max(union, 1))
    return np.nanmean(ious)

def train_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    id2color = {label.id: label.color for label in labels if label.id >= 0}
    n_classes = len([label for label in labels if label.id >= 0])
    batch_size = 12
    
    # Define target size for consistent image dimensions
    target_size = (256, 512)  # (height, width)

    # Load full training dataset
    img_dir = "../data/data_semantics/training/image_2"
    mask_dir = "../data/data_semantics/training/semantic"
    full_dataset = KittiSegmentationDataset(img_dir, mask_dir, augment=False, target_size=target_size)
    
    # Split into train and validation (80/20 split)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    # Use random_split for reproducible splits
    torch.manual_seed(42)  # For reproducibility
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # Create augmented training dataset
    train_set_augmented = KittiSegmentationDataset(img_dir, mask_dir, augment=True, target_size=target_size)
    
    # Create subset for training with augmentation
    train_indices = train_dataset.indices
    train_subset = torch.utils.data.Subset(train_set_augmented, train_indices)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

    print(f"Training samples: {len(train_subset)}")
    print(f"Validation samples: {len(val_dataset)}")

    model = UNet(n_classes=n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255)
    n_epochs = 500

    train_losses, val_losses, val_ious = [], [], []

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('plots', exist_ok=True)

    for epoch in range(1, n_epochs+1):
        model.train()
        train_loss = 0
        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{n_epochs} [Train]"):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)
        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0
        iou = 0
        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch}/{n_epochs} [Val]"):
                imgs, masks = imgs.to(device), masks.to(device)
                out = model(imgs)
                loss = criterion(out, masks)
                val_loss += loss.item() * imgs.size(0)
                preds = torch.argmax(out, dim=1)
                iou += compute_iou(preds, masks, n_classes) * imgs.size(0)
            val_loss /= len(val_loader.dataset)
            iou /= len(val_loader.dataset)
            val_losses.append(val_loss)
            val_ious.append(iou)

        print(f"Epoch {epoch}/{n_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, mIoU: {iou:.4f}")

        # Checkpointing
        if epoch % 10 == 0 or epoch == n_epochs:
            torch.save(model.state_dict(), f'checkpoints/unet_epoch{epoch}.pt')
            # Save plot
            plt.figure(figsize=(10,5))
            plt.plot(train_losses, label="Train Loss")
            plt.plot(val_losses, label="Val Loss")
            plt.plot(val_ious, label="Val mIoU")
            plt.legend()
            plt.title("Losses and mIoU")
            plt.savefig(f'plots/training_plot_epoch{epoch}.png')
            plt.close()
    print("Training Complete!")

if __name__ == '__main__':
    train_model()