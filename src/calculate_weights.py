# calculate_weights.py

import pandas as pd
import numpy as np
import torch
import os
import sys

# Add src folder to path
project_root = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(project_root, 'src'))

from config import PATH_TRAIN_CSV, DISEASE_CLASSES

def calculate_class_weights():
    """
    Compute class weights for multi-label classification based on inverse frequency.
    Useful for handling class imbalance in Focal Loss or BCEWithLogitsLoss.
    """
    print("🔬 Starting calculation of class weights...")
    
    # 1. Load training CSV
    try:
        df = pd.read_csv(PATH_TRAIN_CSV)
    except FileNotFoundError:
        print(f"Error: File '{PATH_TRAIN_CSV}' not found. Check the path.")
        return None

    total_samples = len(df)
    
    # 2. Count positive samples for each class
    class_counts = df[DISEASE_CLASSES].sum()
    
    # 3. Compute inverse-frequency weights
    # If a class has zero positives (should not happen), set count to 1
    class_counts[class_counts == 0] = 1
    
    weights = total_samples / class_counts
    
    # 4. Normalize weights so that their mean equals 1
    normalized_weights = weights / np.mean(weights)
    
    # 5. Convert to PyTorch tensor
    weights_tensor = torch.tensor(normalized_weights.values, dtype=torch.float32)

    print("\nCalculation completed. Class weights:")
    print("-" * 40)
    for disease, weight in zip(DISEASE_CLASSES, normalized_weights):
        print(f"{disease:<20}: {weight:.3f}")
    print("-" * 40)
    
    return weights_tensor

if __name__ == '__main__':
    weights = calculate_class_weights()
    if weights is not None:
        # Save weights for use in training (main.py)
        torch.save(weights, 'class_weights.pth')
        print("\n💾 Class weights saved to 'class_weights.pth'.")
