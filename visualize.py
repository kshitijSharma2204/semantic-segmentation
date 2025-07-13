import matplotlib.pyplot as plt
import numpy as np
import torch

def denormalize(tensor):
    """
    Denormalize tensor that was normalized with ImageNet statistics
    """
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean

def visualize(img, mask, pred_mask=None, id2color=None):
    # Denormalize the image first
    img_denorm = denormalize(img)
    
    # Convert to numpy and ensure proper range [0, 1]
    img_np = img_denorm.permute(1, 2, 0).cpu().numpy()
    img_np = np.clip(img_np, 0, 1)  # Ensure values are in [0, 1] range
    
    mask = mask.cpu().numpy()
    
    if id2color is not None:
        mask_color = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for k, v in id2color.items():
            mask_color[mask == k] = v
    else:
        mask_color = mask

    plt.figure(figsize=(12, 4))
    
    # Plot original image
    plt.subplot(1, 3, 1)
    plt.title("Input Image")
    plt.imshow(img_np)
    plt.axis('off')
    
    # Plot ground truth mask
    plt.subplot(1, 3, 2)
    plt.title("GT Mask")
    plt.imshow(mask_color)
    plt.axis('off')
    
    # Plot predicted mask if provided
    if pred_mask is not None:
        pred_mask = pred_mask.cpu().numpy()
        pred_color = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
        for k, v in id2color.items():
            pred_color[pred_mask == k] = v
        plt.subplot(1, 3, 3)
        plt.title("Predicted Mask")
        plt.imshow(pred_color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def save_visualization(img, mask, pred_mask=None, id2color=None, save_path="visualization.png"):
    """
    Save visualization to file instead of displaying
    """
    # Denormalize the image first
    img_denorm = denormalize(img)
    
    # Convert to numpy and ensure proper range [0, 1]
    img_np = img_denorm.permute(1, 2, 0).cpu().numpy()
    img_np = np.clip(img_np, 0, 1)  # Ensure values are in [0, 1] range
    
    mask = mask.cpu().numpy()
    
    if id2color is not None:
        mask_color = np.zeros((*mask.shape, 3), dtype=np.uint8)
        for k, v in id2color.items():
            mask_color[mask == k] = v
    else:
        mask_color = mask

    plt.figure(figsize=(12, 4))
    
    # Plot original image
    plt.subplot(1, 3, 1)
    plt.title("Input Image")
    plt.imshow(img_np)
    plt.axis('off')
    
    # Plot ground truth mask
    plt.subplot(1, 3, 2)
    plt.title("GT Mask")
    plt.imshow(mask_color)
    plt.axis('off')
    
    # Plot predicted mask if provided
    if pred_mask is not None:
        pred_mask = pred_mask.cpu().numpy()
        pred_color = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
        for k, v in id2color.items():
            pred_color[pred_mask == k] = v
        plt.subplot(1, 3, 3)
        plt.title("Predicted Mask")
        plt.imshow(pred_color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()