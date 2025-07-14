import torch
import os
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import cv2
from tqdm import tqdm
import re
import glob

# Import your existing model and other dependencies
from model import UNet
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.devkit_semantics.devkit.helpers.labels import labels

class FrameSequenceDataset(Dataset):
    """Dataset for loading frame sequences from directory"""
    def __init__(self, frame_dir, target_size=(256, 512)):
        self.frame_dir = frame_dir
        self.target_size = target_size
        
        # Get all image files and sort them numerically
        self.frame_paths = self.get_sorted_frame_paths(frame_dir)
        print(f"Found {len(self.frame_paths)} frames")
        
        self.transform = A.Compose([
            A.Resize(height=target_size[0], width=target_size[1]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    
    def get_sorted_frame_paths(self, frame_dir):
        """Get frame paths sorted numerically"""
        # Get all image files
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
        frame_files = []
        for ext in extensions:
            frame_files.extend(glob.glob(os.path.join(frame_dir, ext)))
        
        # Sort numerically based on the number in filename
        def extract_number(filename):
            # Extract number from filename like "0000000000.png"
            numbers = re.findall(r'\d+', os.path.basename(filename))
            return int(numbers[0]) if numbers else 0
        
        frame_files.sort(key=extract_number)
        return frame_files
    
    def __len__(self):
        return len(self.frame_paths)
    
    def __getitem__(self, idx):
        frame_path = self.frame_paths[idx]
        frame = np.array(Image.open(frame_path).convert("RGB"))
        
        # Apply transformations
        transformed = self.transform(image=frame)
        image = transformed['image']
        
        return image, idx, frame_path

def load_frames_from_sequence(frame_dir):
    """Load frames from numbered sequence"""
    print(f"Loading frames from {frame_dir}...")
    
    # Get all image files and sort them
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
    frame_files = []
    for ext in extensions:
        frame_files.extend(glob.glob(os.path.join(frame_dir, ext)))
    
    # Sort numerically
    def extract_number(filename):
        numbers = re.findall(r'\d+', os.path.basename(filename))
        return int(numbers[0]) if numbers else 0
    
    frame_files.sort(key=extract_number)
    
    # Load frames
    frames = []
    for frame_path in tqdm(frame_files, desc="Loading frames"):
        frame = cv2.imread(frame_path)
        if frame is not None:
            frames.append(frame)
    
    print(f"Loaded {len(frames)} frames")
    return frames

def process_frame_sequence(checkpoint_path, frame_dir, output_dir="sequence_results", 
                         batch_size=8, fps=30):
    """
    Process frame sequence and create video outputs
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    id2color = {label.id: label.color for label in labels if label.id >= 0}
    n_classes = len([label for label in labels if label.id >= 0])
    
    print(f"Processing frame sequence from: {frame_dir}")
    print(f"Device: {device}")
    print(f"Number of classes: {n_classes}")
    
    # Create dataset and loader
    dataset = FrameSequenceDataset(frame_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Load model
    model = UNet(n_classes=n_classes).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    print(f"Loaded model from {checkpoint_path}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get first frame to determine original dimensions
    first_frame = cv2.imread(dataset.frame_paths[0])
    orig_height, orig_width = first_frame.shape[:2]
    
    # Store processed frames
    processed_frames = []
    segmentation_frames = []
    overlay_frames = []
    
    print("Processing frames...")
    with torch.no_grad():
        for batch_imgs, batch_indices, batch_paths in tqdm(loader, desc="Processing batches"):
            batch_imgs = batch_imgs.to(device)
            
            # Get predictions
            preds = model(batch_imgs)
            preds = torch.argmax(preds, dim=1)
            
            # Process each frame in batch
            for i in range(batch_imgs.size(0)):
                img = batch_imgs[i]
                pred = preds[i]
                frame_idx = batch_indices[i].item()
                frame_path = batch_paths[i]
                
                # Load original frame
                orig_frame = cv2.imread(frame_path)
                
                # Create segmentation visualization
                seg_frame = create_segmentation_frame(pred, id2color, orig_width, orig_height)
                
                # Create overlay
                overlay_frame = create_overlay_frame(orig_frame, pred, id2color, alpha=0.6)
                
                # Store frames
                processed_frames.append(orig_frame)
                segmentation_frames.append(seg_frame)
                overlay_frames.append(overlay_frame)
    
    # Create output videos and GIFs
    sequence_name = os.path.basename(frame_dir.rstrip('/'))
    
    print("Creating video outputs...")
    
    # 1. Original video
    orig_video_path = os.path.join(output_dir, f"{sequence_name}_original.mp4")
    save_frames_as_video(processed_frames, orig_video_path, fps)
    
    # 2. Segmentation video
    seg_video_path = os.path.join(output_dir, f"{sequence_name}_segmentation.mp4")
    save_frames_as_video(segmentation_frames, seg_video_path, fps)
    
    # 3. Overlay video
    overlay_video_path = os.path.join(output_dir, f"{sequence_name}_overlay.mp4")
    save_frames_as_video(overlay_frames, overlay_video_path, fps)
    
    # 4. Side-by-side comparison video
    comparison_frames = []
    for orig, seg in zip(processed_frames, segmentation_frames):
        comparison = np.hstack([orig, seg])
        comparison_frames.append(comparison)
    
    comparison_video_path = os.path.join(output_dir, f"{sequence_name}_comparison.mp4")
    save_frames_as_video(comparison_frames, comparison_video_path, fps)
    
    print("Creating GIF outputs...")
    
    # 5. Create GIFs (downsampled for smaller file size)
    gif_fps = min(fps, 10)  # Limit GIF fps to 10 for smaller files
    
    # Original GIF
    orig_gif_path = os.path.join(output_dir, f"{sequence_name}_original.gif")
    save_frames_as_gif(processed_frames, orig_gif_path, gif_fps, max_frames=100)
    
    # Segmentation GIF
    seg_gif_path = os.path.join(output_dir, f"{sequence_name}_segmentation.gif")
    save_frames_as_gif(segmentation_frames, seg_gif_path, gif_fps, max_frames=100)
    
    # Overlay GIF
    overlay_gif_path = os.path.join(output_dir, f"{sequence_name}_overlay.gif")
    save_frames_as_gif(overlay_frames, overlay_gif_path, gif_fps, max_frames=100)
    
    # Comparison GIF
    comparison_gif_path = os.path.join(output_dir, f"{sequence_name}_comparison.gif")
    save_frames_as_gif(comparison_frames, comparison_gif_path, gif_fps, max_frames=100)
    
    print(f"\nProcessing complete! Results saved to: {output_dir}")
    print(f"Videos:")
    print(f"  - Original: {orig_video_path}")
    print(f"  - Segmentation: {seg_video_path}")
    print(f"  - Overlay: {overlay_video_path}")
    print(f"  - Comparison: {comparison_video_path}")
    print(f"GIFs:")
    print(f"  - Original: {orig_gif_path}")
    print(f"  - Segmentation: {seg_gif_path}")
    print(f"  - Overlay: {overlay_gif_path}")
    print(f"  - Comparison: {comparison_gif_path}")

def create_segmentation_frame(pred_mask, id2color, target_width, target_height):
    """Create colored segmentation frame"""
    pred_mask = pred_mask.cpu().numpy()
    
    # Create colored mask
    pred_color = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    for k, v in id2color.items():
        pred_color[pred_mask == k] = v
    
    # Resize to original dimensions
    pred_color = cv2.resize(pred_color, (target_width, target_height))
    
    # Convert RGB to BGR for OpenCV
    pred_color = cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR)
    
    return pred_color

def create_overlay_frame(orig_frame, pred_mask, id2color, alpha=0.6):
    """Create overlay frame with original + segmentation"""
    pred_mask = pred_mask.cpu().numpy()
    
    # Create colored mask
    pred_color = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
    for k, v in id2color.items():
        pred_color[pred_mask == k] = v
    
    # Resize mask to original frame size
    h, w = orig_frame.shape[:2]
    pred_color = cv2.resize(pred_color, (w, h))
    
    # Convert RGB to BGR for OpenCV
    pred_color = cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR)
    
    # Create overlay
    overlay = cv2.addWeighted(orig_frame, 1-alpha, pred_color, alpha, 0)
    
    return overlay

def save_frames_as_video(frames, output_path, fps):
    """Save list of frames as MP4 video"""
    if not frames:
        return
    
    # Get frame dimensions
    h, w = frames[0].shape[:2]
    
    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    for frame in frames:
        out.write(frame)
    
    out.release()
    print(f"Video saved: {output_path}")

def save_frames_as_gif(frames, output_path, fps, max_frames=None, resize_factor=0.5):
    """Save frames as GIF with optional downsampling"""
    if not frames:
        return
    
    # Limit number of frames for smaller GIF
    if max_frames and len(frames) > max_frames:
        step = len(frames) // max_frames
        frames = frames[::step]
    
    # Convert frames to PIL Images
    pil_frames = []
    for frame in frames:
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize for smaller GIF
        if resize_factor != 1.0:
            h, w = frame_rgb.shape[:2]
            new_h, new_w = int(h * resize_factor), int(w * resize_factor)
            frame_rgb = cv2.resize(frame_rgb, (new_w, new_h))
        
        pil_frame = Image.fromarray(frame_rgb)
        pil_frames.append(pil_frame)
    
    # Save as GIF
    duration = int(1000 / fps)  # Duration in milliseconds
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration,
        loop=0,
        optimize=True
    )
    print(f"GIF saved: {output_path}")

def process_streaming_sequence(checkpoint_path, frame_dir, output_video_path, 
                             output_gif_path, fps=30, batch_size=1):
    """
    Memory-efficient streaming processing of frame sequence
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    id2color = {label.id: label.color for label in labels if label.id >= 0}
    n_classes = len([label for label in labels if label.id >= 0])
    
    # Load model
    model = UNet(n_classes=n_classes).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    # Get sorted frame paths
    dataset = FrameSequenceDataset(frame_dir)
    
    # Setup video writer
    first_frame = cv2.imread(dataset.frame_paths[0])
    height, width = first_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # For GIF - collect frames (limit to avoid memory issues)
    gif_frames = []
    max_gif_frames = 100
    frame_step = max(1, len(dataset) // max_gif_frames)
    
    print(f"Processing {len(dataset)} frames with streaming...")
    
    with torch.no_grad():
        for i, (img_tensor, idx, frame_path) in enumerate(tqdm(dataset)):
            # Add batch dimension
            img_tensor = img_tensor.unsqueeze(0).to(device)
            
            # Get prediction
            pred = model(img_tensor)
            pred = torch.argmax(pred, dim=1).squeeze(0)
            
            # Load original frame
            orig_frame = cv2.imread(frame_path)
            
            # Create overlay
            overlay_frame = create_overlay_frame(orig_frame, pred, id2color, alpha=0.6)
            
            # Write to video
            out.write(overlay_frame)
            
            # Collect frames for GIF (downsampled)
            if i % frame_step == 0 and len(gif_frames) < max_gif_frames:
                frame_rgb = cv2.cvtColor(overlay_frame, cv2.COLOR_BGR2RGB)
                # Resize for smaller GIF
                h, w = frame_rgb.shape[:2]
                frame_rgb = cv2.resize(frame_rgb, (w//2, h//2))
                gif_frames.append(Image.fromarray(frame_rgb))
    
    out.release()
    
    # Save GIF
    if gif_frames:
        duration = int(1000 / min(fps, 10))  # Limit GIF fps
        gif_frames[0].save(
            output_gif_path,
            save_all=True,
            append_images=gif_frames[1:],
            duration=duration,
            loop=0,
            optimize=True
        )
    
    print(f"Streaming processing complete!")
    print(f"Video saved: {output_video_path}")
    print(f"GIF saved: {output_gif_path}")

# Usage
if __name__ == "__main__":
    # Configuration
    checkpoint_path = 'checkpoints/unet_epoch500.pt'
    frame_directory = '../2011_09_26_drive_0018_sync/2011_09_26/2011_09_26_drive_0018_sync/image_02/data'
    
    # Method 1: Full processing (creates multiple outputs)
    print("=== Full Frame Sequence Processing ===")
    process_frame_sequence(
        checkpoint_path, 
        frame_directory, 
        output_dir="frame_sequence_results",
        batch_size=8,
        fps=30
    )
    
    # Method 2: Streaming processing (memory efficient, single output)
    print("\n=== Streaming Processing ===")
    process_streaming_sequence(
        checkpoint_path,
        frame_directory,
        output_video_path="streaming_result.mp4",
        output_gif_path="streaming_result.gif",
        fps=30
    )
    
    print("\nAll processing complete!")