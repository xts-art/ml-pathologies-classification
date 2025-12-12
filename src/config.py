# config.py

import torch
import os

# -------------------------------------------------------
# Project Paths
# -------------------------------------------------------

# Path to this config file (src folder)
PATH_ROOT = os.path.dirname(os.path.abspath(__file__))

# Project root directory (one level up from src)
PATH_PROJECT_ROOT = os.path.abspath(os.path.join(PATH_ROOT, os.pardir))

# -------------------------------------------------------
# Device Configuration
# -------------------------------------------------------
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------------
# Image Paths
# -------------------------------------------------------

# Root folder for original images (PNG)
IMG_ROOT = r'C:\Users\tsaya\Downloads'

# Folder for preprocessed PT tensors (used for model training)
PATH_TO_PT_FILES = r'F:\gdtnetpp_project\images\preprocessed_pt'

# Folder with PNG images for visualization purposes
PATH_TO_PNG_FILES = os.path.join(IMG_ROOT, 'images')

# -------------------------------------------------------
# CSV Files (Train / Validation Labels)
# -------------------------------------------------------
PATH_TRAIN_CSV = os.path.join(PATH_PROJECT_ROOT, 'miccai2023_nih-cxr-lt_labels_train.csv')
PATH_VAL_CSV   = os.path.join(PATH_PROJECT_ROOT, 'miccai2023_nih-cxr-lt_labels_val.csv')

# -------------------------------------------------------
# Model and Checkpoint Paths
# -------------------------------------------------------
PATH_BEST_MODEL  = os.path.join(PATH_PROJECT_ROOT, 'best_chest_xray_model.pth')
PATH_CHECKPOINT  = os.path.join(PATH_PROJECT_ROOT, 'training_checkpoint.pth')

# -------------------------------------------------------
# Training Hyperparameters
# -------------------------------------------------------
NUM_CLASSES   = 15
BATCH_SIZE    = 4
LEARNING_RATE = 1e-5
NUM_EPOCHS    = 100

# -------------------------------------------------------
# Disease Classes (Multi-label Classification)
# -------------------------------------------------------
DISEASE_CLASSES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
    'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule',
    'Pleural Thickening', 'Pneumonia', 'Pneumothorax', 'Pneumoperitoneum'
]

# -------------------------------------------------------
# Ensure necessary directories exist
# -------------------------------------------------------
os.makedirs(IMG_ROOT, exist_ok=True)
os.makedirs(PATH_TO_PT_FILES, exist_ok=True)
os.makedirs(PATH_PROJECT_ROOT, exist_ok=True)
