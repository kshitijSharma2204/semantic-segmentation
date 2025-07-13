import torch
import os
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from model import UNet

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.devkit_semantics.devkit.helpers.labels import labels
from visualize import save_visualization
import matplotlib.pyplot as plt

class TestDataset(Dataset):
    """Dataset for inference - only RGB images, no masks"""
    def __init__(self, image_dir, target_size=(256, 512)):
        self.image_dir = image_dir
        self.target_size = target_size
        self.images = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        self.transform = A.Compose([
            A.Resize(height=target_size[0], width=target_size[1]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = np.array(Image.open(img_path).convert("RGB"))
        
        # Apply resize and normalization
        transformed = self.transform(image=image)
        image = transformed['image']
        
        return image, self.images[idx]

def semantic_segmentation_inference(checkpoint_path, image_dir, output_dir="segmentation_results"):
    """
    Run semantic segmentation on RGB images without ground truth masks
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    id2color = {label.id: label.color for label in labels if label.id >= 0}
    n_classes = len([label for label in labels if label.id >= 0])
    
    print(f"Running semantic segmentation on images in: {image_dir}")
    print(f"Device: {device}")
    print(f"Number of classes: {n_classes}")
    
    # Create dataset and loader
    dataset = TestDataset(image_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    print(f"Found {len(dataset)} images for segmentation")
    
    # Load model
    model = UNet(n_classes=n_classes).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "visualizations"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "colored_masks"), exist_ok=True)
    
    with torch.no_grad():
        for idx, (imgs, filenames) in enumerate(loader):
            imgs = imgs.to(device)
            
            # Get predictions
            preds = model(imgs)
            preds = torch.argmax(preds, dim=1)
            
            # Process each image in batch
            for i in range(imgs.size(0)):
                img = imgs[i]
                pred = preds[i]
                filename = filenames[i]
                base_name = os.path.splitext(filename)[0]
                
                print(f"Processing {idx+1}/{len(dataset)}: {filename}")
                
                # Save visualization (input + prediction)
                viz_path = os.path.join(output_dir, "visualizations", f"{base_name}_result.png")
                save_segmentation_visualization(img, pred, id2color, viz_path)
                
                # Save raw prediction mask (grayscale)
                mask_path = os.path.join(output_dir, "masks", f"{base_name}_mask.png")
                save_prediction_mask(pred, mask_path)
                
                # Save colored prediction mask
                colored_mask_path = os.path.join(output_dir, "colored_masks", f"{base_name}_colored.png")
                save_colored_mask(pred, id2color, colored_mask_path)
    
    print(f"\nSegmentation complete! Results saved to: {output_dir}")
    print(f"- Visualizations: {output_dir}/visualizations/")
    print(f"- Raw masks: {output_dir}/masks/")
    print(f"- Colored masks: {output_dir}/colored_masks/")

def save_segmentation_visualization(img, pred_mask, id2color, save_path):
    """
    Save visualization with input image and predicted segmentation
    """
    # Denormalize the image - move tensors to same device as img
    device = img.device
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
    img_denorm = img * std + mean
    
    # Convert to numpy
    img_np = img_denorm.permute(1, 2, 0).cpu().numpy()
    img_np = np.clip(img_np, 0, 1)
    
    pred_mask = pred_mask.cpu().numpy()
    
    # Create colored mask
    pred_color = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    for k, v in id2color.items():
        pred_color[pred_mask == k] = v
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    
    # Original image
    plt.subplot(1, 2, 1)
    plt.title("Input Image")
    plt.imshow(img_np)
    plt.axis('off')
    
    # Predicted segmentation
    plt.subplot(1, 2, 2)
    plt.title("Semantic Segmentation")
    plt.imshow(pred_color)
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def save_prediction_mask(pred_mask, save_path):
    """
    Save raw prediction mask as grayscale image
    """
    pred_np = pred_mask.cpu().numpy().astype(np.uint8)
    Image.fromarray(pred_np).save(save_path)

def save_colored_mask(pred_mask, id2color, save_path):
    """
    Save colored prediction mask
    """
    pred_mask = pred_mask.cpu().numpy()
    pred_color = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    
    for k, v in id2color.items():
        pred_color[pred_mask == k] = v
    
    Image.fromarray(pred_color).save(save_path)

def create_overlay_visualization(checkpoint_path, image_dir, output_dir="overlay_results", alpha=0.6):
    """
    Create overlay visualizations (original image + transparent segmentation)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    id2color = {label.id: label.color for label in labels if label.id >= 0}
    n_classes = len([label for label in labels if label.id >= 0])
    
    dataset = TestDataset(image_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    model = UNet(n_classes=n_classes).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for idx, (imgs, filenames) in enumerate(loader):
            imgs = imgs.to(device)
            preds = model(imgs)
            preds = torch.argmax(preds, dim=1)
            
            for i in range(imgs.size(0)):
                img = imgs[i]
                pred = preds[i]
                filename = filenames[i]
                base_name = os.path.splitext(filename)[0]
                
                # Denormalize image - fix device compatibility here too
                device = img.device
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
                img_denorm = img * std + mean
                img_np = img_denorm.permute(1, 2, 0).cpu().numpy()
                img_np = np.clip(img_np, 0, 1)
                
                # Create colored mask
                pred_mask = pred.cpu().numpy()
                pred_color = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
                for k, v in id2color.items():
                    pred_color[pred_mask == k] = v
                pred_color = pred_color.astype(np.float32) / 255.0
                
                # Create overlay
                overlay = (1 - alpha) * img_np + alpha * pred_color
                
                # Save overlay
                plt.figure(figsize=(10, 6))
                plt.imshow(overlay)
                plt.axis('off')
                plt.title(f"Segmentation Overlay: {filename}")
                
                overlay_path = os.path.join(output_dir, f"{base_name}_overlay.png")
                plt.savefig(overlay_path, dpi=150, bbox_inches='tight')
                plt.close()
    
    print(f"Overlay visualizations saved to: {output_dir}")

# Usage
if __name__ == "__main__":
    # Configuration
    checkpoint_path = 'checkpoints/unet_epoch500.pt'
    test_image_dir = "../data/data_semantics/testing/image_2"
    
    # Run semantic segmentation
    print("=== Running Semantic Segmentation ===")
    semantic_segmentation_inference(checkpoint_path, test_image_dir)
    
    # Create overlay visualizations
    print("\n=== Creating Overlay Visualizations ===")
    create_overlay_visualization(checkpoint_path, test_image_dir)
    
    print("\nAll done! Check the output folders for results.")