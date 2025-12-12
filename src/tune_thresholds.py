# tune_thresholds.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import json
import os
import sys

from sklearn.metrics import f1_score, roc_auc_score

# --- Reliable import of project modules ---
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(project_root, 'src'))

from config import (
    NUM_CLASSES, DEVICE, PATH_VAL_CSV, PATH_TO_PT_FILES,
    PATH_BEST_MODEL, BATCH_SIZE, DISEASE_CLASSES
)
from training_functions import MultiLabelClassifier
from dataset import ChestXrayDataset, get_dataloaders

# Path for saving optimal thresholds
PATH_THRESHOLDS = os.path.join(os.path.dirname(PATH_BEST_MODEL), 'optimal_thresholds.json')


def get_predictions(model, dataloader, device):
    """
    Collect raw model outputs (logits) and true labels for the entire dataset.
    Returns:
        all_outputs: ndarray of logits
        all_targets: ndarray of ground-truth labels
    """
    model.eval()

    all_targets = []
    all_outputs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)

            all_targets.append(labels.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())

    return np.concatenate(all_outputs), np.concatenate(all_targets)


def tune_thresholds():
    """
    Load the best model and compute optimal per-class thresholds
    by maximizing F1-score on the validation set.
    """
    print("--- 1. Initializing and Loading Validation Data ---")

    _, val_loader = get_dataloaders()

    print("--- 2. Loading Best Model ---")

    if not os.path.exists(PATH_BEST_MODEL):
        print(f"Error: Best model not found at: {PATH_BEST_MODEL}")
        print("Train the model first using main.py.")
        return

    model = MultiLabelClassifier(NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(PATH_BEST_MODEL, map_location=DEVICE))
    model.eval()

    print(f"Model successfully loaded. Device: {DEVICE}")

    # Step 3: Compute model predictions
    print("--- 3. Generating Validation Predictions ---")
    outputs, targets = get_predictions(model, val_loader, DEVICE)

    # Convert logits to probabilities using sigmoid
    probabilities = 1 / (1 + np.exp(-outputs))

    print("--- 4. Searching for Optimal Thresholds (F1 Maximization) ---")

    optimal_thresholds = {}

    for i, class_name in enumerate(DISEASE_CLASSES):
        best_f1 = -1.0
        best_threshold = 0.5  # default starting point

        # Sweep thresholds from 0.1 to 0.9
        for threshold in np.arange(0.1, 0.91, 0.01):
            binary_preds = (probabilities[:, i] > threshold).astype(int)

            try:
                f1 = f1_score(targets[:, i], binary_preds, zero_division=0)

                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold

            except ValueError:
                # Occurs if the class is missing in the dataset
                continue

        optimal_thresholds[class_name] = round(best_threshold, 3)
        print(f"[{i+1}/{NUM_CLASSES}] {class_name:<20}: "
              f"F1 = {best_f1:.4f}, Threshold = {best_threshold:.3f}")

    # Step 5: Save thresholds to JSON
    with open(PATH_THRESHOLDS, 'w') as f:
        json.dump(optimal_thresholds, f, indent=4)

    print("\n" + "=" * 60)
    print(f"Optimal thresholds saved to: {PATH_THRESHOLDS}")
    print("Use these thresholds for final model evaluation.")
    print("=" * 60)


if __name__ == '__main__':
    tune_thresholds()
