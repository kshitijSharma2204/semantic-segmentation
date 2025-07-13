import os
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

class KittiSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, augment=False, target_size=(256, 512)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(image_dir))
        self.masks = sorted(os.listdir(mask_dir))
        self.augment = augment
        self.target_size = target_size
        
        # Augmentation transforms
        self.aug = A.Compose([
                   A.Resize(height=target_size[0], width=target_size[1]),
                   A.HorizontalFlip(p=0.5),
                   A.RandomBrightnessContrast(p=0.3),
                   A.ColorJitter(p=0.3),
                   A.GaussianBlur(p=0.2),
                   A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.3),
                   A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                   ToTensorV2()
        ])
        
        
        self.no_aug = A.Compose([
            A.Resize(height=target_size[0], width=target_size[1]),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.images)   

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))

        if self.augment:
            augmented = self.aug(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']
        else:
            # Apply resize and normalization
            augmented = self.no_aug(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']

        mask = mask.long()
        return image, mask