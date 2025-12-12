# main.py

import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

# ---------------------------------------------------------------------
#                       Safe Imports (Project Structure)
# ---------------------------------------------------------------------
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(project_root, 'src'))
sys.path.append(project_root)

from config import (
    NUM_CLASSES, DEVICE, LEARNING_RATE, NUM_EPOCHS,
    PATH_BEST_MODEL, PATH_CHECKPOINT, DISEASE_CLASSES
)

from training_functions import (
    MultiLabelClassifier, train_epoch, validate_epoch,
    save_checkpoint, load_checkpoint, evaluate_model
)

from dataset import get_dataloaders
from focal_loss import FocalLoss


# Path to the file containing class weights for Focal Loss
PATH_CLASS_WEIGHTS = "class_weights.pth"


# =====================================================================
#                               Main
# =====================================================================

def main():

    print(f"--- 1. Initializing Data (LR = {LEARNING_RATE}) ---")

    # ---------------------------------------------------------
    # 1. Load training and validation dataloaders
    # ---------------------------------------------------------
    try:
        train_loader, val_loader = get_dataloaders()
    except Exception as e:
        print(f"Error initializing dataloaders: {e}")
        return

    print(f"Data loaded successfully. "
          f"Train samples: {len(train_loader.dataset)}, "
          f"Val samples: {len(val_loader.dataset)}")

    # ---------------------------------------------------------
    # 2. Initialize model, optimizer and loss function
    # ---------------------------------------------------------
    model = MultiLabelClassifier(NUM_CLASSES).to(DEVICE)

    # AdamW with weight decay for better generalization
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    print("Using AdamW optimizer (weight_decay=1e-3).")

    # Load class weights (pos_weight) for Focal Loss
    try:
        class_weights = torch.load(PATH_CLASS_WEIGHTS).to(DEVICE)
        print("Class weights loaded.")
    except FileNotFoundError:
        print(f"Error: '{PATH_CLASS_WEIGHTS}' not found. Run calculate_weights.py first.")
        return

    # FocalLoss with class balancing
    criterion = FocalLoss(gamma=3.0, pos_weight=class_weights)
    print("Using Focal Loss (gamma=3.0) with class balancing.")

    start_epoch = 1
    best_val_loss = float("inf")
    best_mean_auc = -1.0

    # ---------------------------------------------------------
    # Learning Rate Scheduler (ReduceLROnPlateau on AUC)
    # ---------------------------------------------------------
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.1, patience=3, verbose=True
    )
    print("Using LR scheduler ReduceLROnPlateau (mode='max' on AUC).")

    # ---------------------------------------------------------
    # 3. Optionally load checkpoint if it exists
    # ---------------------------------------------------------
    if os.path.exists(PATH_CHECKPOINT):
        start_epoch, best_val_loss, best_mean_auc = load_checkpoint(
            PATH_CHECKPOINT, model, optimizer, DEVICE
        )
        print(f"Checkpoint loaded. Resuming from epoch {start_epoch}. "
              f"Best AUC so far: {best_mean_auc:.4f}")
    else:
        print("Starting training from scratch (Focal Loss enabled).")

    print(f"\nStarting training for {NUM_EPOCHS} epochs "
          f"(starting from epoch {start_epoch})...\n")

    # =================================================================
    #                          Training Loop
    # =================================================================
    for epoch in range(start_epoch, NUM_EPOCHS + 1):

        epoch_start_time = time.time()

        # -------------------------------
        # 1. Training
        # -------------------------------
        train_loss = train_epoch(model, train_loader, criterion, optimizer, DEVICE)

        # -------------------------------
        # 2. Validation Loss
        # -------------------------------
        val_loss = validate_epoch(model, val_loader, criterion, DEVICE)

        # -------------------------------
        # 3. Evaluation (ROC AUC)
        # -------------------------------
        val_metrics = evaluate_model(model, val_loader, DEVICE, DISEASE_CLASSES)
        mean_auc = val_metrics["mean_auc"]

        # -------------------------------
        # 4. Update LR Scheduler
        # -------------------------------
        scheduler.step(mean_auc)

        # -------------------------------
        # 5. Epoch Summary
        # -------------------------------
        print("\n" + "-" * 45)
        print(f"--- Epoch {epoch} Summary ---  "
              f"| Time: {time.time() - epoch_start_time:.2f} sec")
        print(f"  Training Loss (Focal):  {train_loss:.4f}")
        print(f"  Validation Loss (Focal): {val_loss:.4f}")
        print(f"  Mean ROC AUC:            {mean_auc:.4f}")
        print("-" * 45)

        # -------------------------------
        # 6. Save checkpoint (every epoch)
        # -------------------------------
        save_checkpoint(
            epoch + 1, model, optimizer,
            val_loss, best_val_loss, best_mean_auc,
            PATH_CHECKPOINT
        )

        # -------------------------------
        # 7. Save Best Model (based on AUC)
        # -------------------------------
        if mean_auc > best_mean_auc:
            print(f"New best model! AUC improved: {best_mean_auc:.4f} → {mean_auc:.4f}")
            best_mean_auc = mean_auc
            torch.save(model.state_dict(), PATH_BEST_MODEL)

        # Update best_val_loss only for bookkeeping
        if val_loss < best_val_loss:
            best_val_loss = val_loss

    # =================================================================
    #                       Training Finished
    # =================================================================
    print("\nTraining completed!")
    print(f"Best model saved to '{PATH_BEST_MODEL}' with AUC = {best_mean_auc:.4f}")


if __name__ == "__main__":
    main()
