import os
import cv2
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class KittiSegmentationDataset(Dataset):
    """KITTI semantics with the masks mapped from label IDs to the 19 trainIds (255 = ignore).

    Each image is resized once to `size` (352x1184 keeps KITTI's ~3.3:1 aspect ratio).
    With cache=True all of them are decoded up front and kept in RAM, which is about
    250 MB for the 200 training images, so an epoch doesn't decode PNGs again.
    """
    def __init__(self, image_dir, mask_dir, names, lut, size=(352, 1184), crop=(352, 592),
                 scale=(0.75, 1.5), augment=True, cache=True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = list(names)
        self.lut = lut
        self.size = size
        self.augment = augment
        self.cache = [self.load(name) for name in self.images] if cache else None

        self.aug = A.Compose([
            A.RandomScale(scale_limit=(scale[0] - 1, scale[1] - 1), p=1.0),
            A.PadIfNeeded(min_height=crop[0], min_width=crop[1], border_mode=cv2.BORDER_CONSTANT,
                          fill=0, fill_mask=255),
            A.RandomCrop(height=crop[0], width=crop[1]),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.ColorJitter(p=0.3),
            A.GaussianBlur(p=0.2),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2()
        ])
        self.no_aug = A.Compose([A.Normalize(mean=MEAN, std=STD), ToTensorV2()])

    def load(self, name):
        image = np.array(Image.open(os.path.join(self.image_dir, name)).convert("RGB"))
        mask = self.lut[np.array(Image.open(os.path.join(self.mask_dir, name)))]
        h, w = self.size
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return image, mask

    def reseed(self, seed):
        # albumentations keeps its own RNG, and forked workers would all start from
        # the same copy of it, so each worker reseeds from its torch seed
        self.aug.set_random_seed(seed)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image, mask = self.cache[idx] if self.cache is not None else self.load(self.images[idx])
        out = (self.aug if self.augment else self.no_aug)(image=image, mask=mask)
        return out['image'], out['mask'].long()
