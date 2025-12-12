# src/scripts/preprocess_all.py

import sys
import os
from tqdm import tqdm
import torch
from PIL import Image
import torchvision.transforms as transforms
import cv2
import numpy as np
from typing import Tuple

# --- Robust path setup for importing config.py ---
# This ensures imports work regardless of where the script is executed.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

try:
    from config import IMG_ROOT, PATH_TO_PT_FILES
except ImportError:
    print("Error: Cannot find config.py. Make sure it exists in the 'src' folder.")
    sys.exit(1)

# =============================================================================
#                         Image Preprocessing Functions
# =============================================================================

def apply_clahe_resize_to_tensor(pil_img: Image.Image, output_size: Tuple[int, int] = (1024, 1024)) -> torch.Tensor:
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization), 
    resizes the image, and converts it to a PyTorch tensor.

    Args:
        pil_img (Image.Image): The input grayscale image in PIL format.
        output_size (Tuple[int, int]): The target output size (width, height).

    Returns:
        torch.Tensor: The preprocessed image tensor (C, H, W).
    """
    # Convert PIL Image (grayscale) to NumPy array (uint8)
    img_np = np.array(pil_img, dtype=np.uint8)
    
    # 1. Apply CLAHE (improves local contrast in chest X-rays)
    # clipLimit=2.0 and tileGridSize=(8, 8) are common parameters
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img_np = clahe.apply(img_np)
    
    # 2. Convert the processed NumPy array back to PIL
    clahe_pil_img = Image.fromarray(clahe_img_np)
    
    # 3. Apply PyTorch transforms (Resize and ToTensor)
    transform_pipeline = transforms.Compose([
        # Use BILINEAR for smooth resizing of the image data
        transforms.Resize(output_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor() # Converts to torch.FloatTensor and scales pixels to [0.0, 1.0]
    ])
    
    tensor_output = transform_pipeline(clahe_pil_img)
    return tensor_output

# =============================================================================
#                             Main Execution Script
# =============================================================================

def run_preprocessing():
    """
    Main function to orchestrate the image preprocessing workflow:
    1. Reads original PNG files.
    2. Applies CLAHE and resizing.
    3. Saves the results as PyTorch tensor (.pt) files.
    """
    # Source directory for original PNG images
    src_dir = os.path.join(IMG_ROOT, "images")
    # Target directory for saving preprocessed .pt files
    save_dir = PATH_TO_PT_FILES
    
    # Ensure the output directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Get a list of all PNG files in the source directory
    image_files = sorted([f for f in os.listdir(src_dir) if f.endswith(".png")])
    
    if not image_files:
        print(f"Error: No PNG files found in {src_dir}")
        return

    print(f"Found {len(image_files)} images for preprocessing.")
    
    # Process files with a progress bar
    for fname in tqdm(image_files, desc="Preprocessing images"):
        path = os.path.join(src_dir, fname)
        
        # --- File Reading ---
        try:
            # Open the image and convert to Grayscale ('L') immediately
            pil_img = Image.open(path).convert("L")
        except Exception as e:
            tqdm.write(f"Failed to read {fname}: {e}")
            continue

        # --- Image Processing ---
        try:
            img_tensor = apply_clahe_resize_to_tensor(pil_img)
        except Exception as e:
            tqdm.write(f"Failed to process {fname}: {e}")
            continue

        # --- Saving the Tensor ---
        try:
            # Create save path (e.g., 'image_id.png' -> 'image_id.pt')
            base_name = os.path.splitext(fname)[0]
            save_path = os.path.join(save_dir, f"{base_name}.pt")
            torch.save(img_tensor, save_path)
        except Exception as e:
            tqdm.write(f"Failed to save tensor for {fname}: {e}")
            continue
    
    print("\nAll available images processed and saved as .pt files!")

if __name__ == '__main__':
    run_preprocessing()