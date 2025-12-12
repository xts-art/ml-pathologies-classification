# dataset.py

import os
import math
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms

from config import (
    PATH_TO_PT_FILES,
    PATH_TRAIN_CSV,
    PATH_VAL_CSV,
    BATCH_SIZE,
    DISEASE_CLASSES
)

# =====================================================================
#                           SimpleCutout
# =====================================================================

class SimpleCutout(object):
    """
    Simple and safe Cutout augmentation for PyTorch tensors.
    Removes a random square region and fills it with the tensor's mean value.
    """
    def __init__(self, max_size=64, p=0.7):
        """
        Args:
            max_size (int): Maximum cutout square side.
            p (float): Probability of applying Cutout.
        """
        self.max_size = max_size
        self.p = p

    def __call__(self, img):
        # Skip augmentation with probability (1 − p)
        if torch.rand(1).item() > self.p:
            return img

        h, w = img.shape[-2], img.shape[-1]

        # Minimal allowed cutout size
        desired_min_size = max(10, w // 8)
        max_rand_limit = self.max_size

        # Safe lower bound for randint
        min_rand_size = min(desired_min_size, max_rand_limit - 1)

        # If Cutout is impossible (too small sizes), skip
        if min_rand_size <= 0:
            return img

        # Random cutout size
        size = torch.randint(min_rand_size, max_rand_limit, (1,)).item()

        # Random center
        y = torch.randint(0, h, (1,)).item()
        x = torch.randint(0, w, (1,)).item()

        # Compute boundaries safely
        y1 = np.clip(y - size // 2, 0, h)
        y2 = np.clip(y + size // 2, 0, h)
        x1 = np.clip(x - size // 2, 0, w)
        x2 = np.clip(x + size // 2, 0, w)

        # Fill value = mean intensity to avoid sharp artificial quadrants
        fill_value = img.mean()

        img[:, y1:y2, x1:x2] = fill_value
        return img


# =====================================================================
#                     ChestXrayDataset Class
# =====================================================================

class ChestXrayDataset(Dataset):
    """
    Dataset for pre-extracted .pt tensor images with multi-label disease annotations.
    """
    def __init__(self, csv_file, root_dir, transform=None, classes=DISEASE_CLASSES):
        self.labels_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.classes = classes

        # Verify class presence
        if not all(cls in self.labels_frame.columns for cls in self.classes):
            raise ValueError("Some required disease classes are missing in the CSV file.")

        # Extract label matrix
        self.labels = self.labels_frame[self.classes].values.astype(np.float32)

        # Replace PNG names with .pt files
        self.labels_frame['id'] = self.labels_frame['id'].apply(
            lambda x: x.replace('.png', '.pt') if x.endswith('.png') else x
        )

    def __len__(self):
        return len(self.labels_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root_dir, self.labels_frame.iloc[idx]['id'])

        # Load precomputed tensor
        image_tensor = torch.load(img_path)

        # Load labels
        label_tensor = torch.tensor(self.labels[idx], dtype=torch.float32)

        # Apply augmentations if any
        if self.transform:
            image_tensor = self.transform(image_tensor)

        return image_tensor, label_tensor


# =====================================================================
#                     Sample Weight Calculation
# =====================================================================

def get_sample_weights(df, classes):
    """
    Computes per-sample weights based on class frequency (inverse frequency weighting).
    Used for WeightedRandomSampler to balance rare diseases.
    """
    class_counts = df[classes].sum().to_numpy()
    class_weights_inv = 1.0 / (class_counts + 1e-5)

    labels_matrix = df[classes].to_numpy()
    sample_weights = labels_matrix @ class_weights_inv

    # Normalize for stability
    sample_weights /= np.mean(sample_weights)

    return sample_weights


# =====================================================================
#                        DataLoader Factory
# =====================================================================

def get_dataloaders():
    """
    Returns dataloaders for training and validation with balanced sampling.
    """

    # -------------------- Transforms --------------------
    train_transform = transforms.Compose([
        transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1)
        ),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        SimpleCutout(max_size=64, p=0.7),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    val_transform = transforms.Compose([
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # -------------------- Datasets --------------------
    train_dataset = ChestXrayDataset(
        csv_file=PATH_TRAIN_CSV,
        root_dir=PATH_TO_PT_FILES,
        transform=train_transform
    )

    val_dataset = ChestXrayDataset(
        csv_file=PATH_VAL_CSV,
        root_dir=PATH_TO_PT_FILES,
        transform=val_transform
    )

    # ---------------- Weighted Sampling ----------------
    sample_weights = get_sample_weights(train_dataset.labels_frame, DISEASE_CLASSES)
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(train_dataset),
        replacement=True
    )

    # ------------------- DataLoaders -------------------
    num_workers = os.cpu_count() // 2 if os.cpu_count() and os.cpu_count() > 1 else 1

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader
