# training_functions.py

import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from torchvision.models import densenet121, DenseNet121_Weights

from config import NUM_CLASSES, DEVICE


# =====================================================================
#                           Model Definition
# =====================================================================

class MultiLabelClassifier(nn.Module):
    """
    Multi-label classifier based on pretrained DenseNet-121.
    Uses a custom head with dropout to reduce overfitting.
    """
    def __init__(self, num_classes):
        super(MultiLabelClassifier, self).__init__()

        # Load ImageNet-pretrained DenseNet-121
        self.densenet = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)

        # Replace classifier with Dropout + Linear(num_ftrs → num_classes)
        num_ftrs = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Sequential(
            nn.Dropout(p=0.5),                 # index 0 — added dropout layer
            nn.Linear(num_ftrs, num_classes)   # index 1 — classifier
        )

    def forward(self, x):
        # Model expects RGB; repeat grayscale channel 3 times
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.densenet(x)


# =====================================================================
#                 Training & Validation (with TQDM)
# =====================================================================

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    dataloader_tqdm = tqdm(dataloader, desc="Training", unit="batch", leave=False)

    for inputs, labels in dataloader_tqdm:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        dataloader_tqdm.set_postfix(loss=f'{loss.item():.4f}')

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0

    dataloader_tqdm = tqdm(dataloader, desc="Validation", unit="batch", leave=False)

    with torch.no_grad():
        for inputs, labels in dataloader_tqdm:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            dataloader_tqdm.set_postfix(loss=f'{loss.item():.4f}')

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss


# =====================================================================
#                 Evaluation (ROC AUC per class)
# =====================================================================

def evaluate_model(model, dataloader, device, classes):
    """
    Computes ROC AUC per class and macro mean AUC.
    """
    model.eval()
    all_targets = []
    all_outputs = []

    dataloader_tqdm = tqdm(dataloader, desc="Evaluating", unit="batch", leave=False)

    with torch.no_grad():
        for inputs, labels in dataloader_tqdm:
            inputs = inputs.to(device)

            logits = model(inputs)
            outputs = torch.sigmoid(logits).cpu().numpy()

            all_outputs.append(outputs)
            all_targets.append(labels.cpu().numpy())

    targets = np.concatenate(all_targets, axis=0)
    outputs = np.concatenate(all_outputs, axis=0)

    auc_scores = []
    for i in range(targets.shape[1]):
        # ROC AUC is undefined if only one class is present
        if len(np.unique(targets[:, i])) > 1:
            auc = roc_auc_score(targets[:, i], outputs[:, i])
            auc_scores.append(auc)
        else:
            auc_scores.append(np.nan)

    mean_auc = np.nanmean(auc_scores)

    return {
        "mean_auc": mean_auc,
        "auc_per_class": dict(zip(classes, auc_scores))
    }


# =====================================================================
#                        Checkpoint Handling
# =====================================================================

def save_checkpoint(epoch, model, optimizer, val_loss, best_val_loss, best_mean_auc, filename):
    """
    Saves model, optimizer state, and metrics including best AUC so far.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'best_val_loss': best_val_loss,
        'best_mean_auc': best_mean_auc
    }
    torch.save(checkpoint, filename)


def load_checkpoint(filename, model, optimizer, device):
    """
    Loads a checkpoint and automatically fixes key mismatches caused
    by modifying the classifier architecture (Dropout insertion).
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Checkpoint not found: {filename}")

    print(f"Loading checkpoint '{filename}'...")
    checkpoint = torch.load(filename, map_location=device)
    state_dict = checkpoint['model_state_dict']

    # Handle old checkpoints (before dropout was added)
    old_weight_key = 'densenet.classifier.weight'
    old_bias_key   = 'densenet.classifier.bias'

    if old_weight_key in state_dict and 'densenet.classifier.1.weight' not in state_dict:
        print("Detected legacy checkpoint format. Renaming classifier keys...")

        state_dict['densenet.classifier.1.weight'] = state_dict.pop(old_weight_key)
        state_dict['densenet.classifier.1.bias']   = state_dict.pop(old_bias_key)

    # Load model and optimizer
    model.load_state_dict(state_dict)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    start_epoch     = checkpoint['epoch']
    best_val_loss   = checkpoint['best_val_loss']
    best_mean_auc   = checkpoint.get('best_mean_auc', -1.0)  # backwards compatibility

    print("Checkpoint loaded successfully.")
    return start_epoch, best_val_loss, best_mean_auc
