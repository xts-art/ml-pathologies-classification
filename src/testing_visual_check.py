# testing_visual_check.py

import os
import sys
import json
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import transforms

# --- Reliable import of project modules ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'src'))

from config import NUM_CLASSES, DEVICE, PATH_BEST_MODEL, DISEASE_CLASSES
from training_functions import MultiLabelClassifier
from dataset import ChestXrayDataset

# Path for loading optimal thresholds (assumed to be next to the best model)
PATH_THRESHOLDS = os.path.join(os.path.dirname(os.path.abspath(PATH_BEST_MODEL)), 'optimal_thresholds.json')


# -------------------------------------------------------------------
# Helper: Load single data point
# -------------------------------------------------------------------
def get_single_data_point(index):
    """
    Load a single data point from PT tensor (for model) and corresponding PNG (for visualization).
    Returns:
        image_tensor: normalized tensor for model input
        labels_tensor: ground truth labels
        visual_image_pil: PIL image for display
        png_filename: filename of the PNG
        dataset_size: total number of samples
    """
    # Assuming config is properly imported and contains paths
    from config import PATH_VAL_CSV, PATH_TO_PT_FILES, PATH_TO_PNG_FILES

    transform = transforms.Compose([
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    dataset = ChestXrayDataset(csv_file=PATH_VAL_CSV,
                               root_dir=PATH_TO_PT_FILES,
                               transform=transform)

    # PT tensor and labels
    image_tensor, labels_tensor = dataset[index]
    image_id_pt = dataset.labels_frame.iloc[index]['id']

    # Corresponding PNG file
    png_filename = image_id_pt.replace('.pt', '.png')
    png_path = os.path.join(PATH_TO_PNG_FILES, png_filename)
    # Convert 'L' (grayscale) for consistent display
    visual_image_pil = Image.open(png_path).convert('L') 

    return image_tensor, labels_tensor, visual_image_pil, png_filename, len(dataset)


# -------------------------------------------------------------------
# Visualization of model prediction with per-class thresholds (with score)
# -------------------------------------------------------------------
def visualize_prediction(sample_index, optimal_thresholds):
    """
    Visualize model prediction and ground truth for a single sample.
    Applies individual thresholds per class and displays the accuracy score.
    """
    image_tensor_pt, true_labels, visual_image_pil, image_filename, _ = get_single_data_point(sample_index)

    # Load model
    model = MultiLabelClassifier(NUM_CLASSES).to(DEVICE)
    try:
        model.load_state_dict(torch.load(PATH_BEST_MODEL, map_location=DEVICE))
    except FileNotFoundError:
        print(f"Error: Model file '{PATH_BEST_MODEL}' not found. Train the model first.")
        return
    model.eval()

    # Prepare input for model (batch dim + repeat channel if needed for 3-channel input)
    single_input = image_tensor_pt.unsqueeze(0).to(DEVICE)
    if single_input.size(1) == 1:
        # Repeat the single channel (grayscale) three times to mimic an RGB image
        single_input_for_model = single_input.repeat(1, 3, 1, 1) 
    else:
        single_input_for_model = single_input

    # Forward pass
    with torch.no_grad():
        logits = model(single_input_for_model)
        probabilities = torch.sigmoid(logits).squeeze().cpu().numpy()

    # Apply per-class thresholds
    threshold_array = np.array([
        optimal_thresholds.get(disease, 0.5)
        for disease in DISEASE_CLASSES
    ])
    predicted_labels = (probabilities > threshold_array).astype(int)

    # --- In Counter ---
    correct_predictions = 0
    total_classes = len(DISEASE_CLASSES)
    # -------------------------------

    # Visualization setup
    fig, ax = plt.subplots(1, 2, figsize=(18, 9))

    # Left panel: PNG image
    ax[0].imshow(visual_image_pil, cmap='gray')
    ax[0].set_title(f"PNG Image: {image_filename}\n(Validation Index: {sample_index})", fontsize=14)
    ax[0].axis('off')

    # Right panel: Prediction vs Ground Truth
    true_text = "GROUND TRUTH:\n"
    for i, disease in enumerate(DISEASE_CLASSES):
        if true_labels[i].item() == 1:
            true_text += f" [V] {disease}\n"

    pred_text = "MODEL PREDICTIONS (PER-CLASS THRESHOLDS):\n"
    for i, disease in enumerate(DISEASE_CLASSES):
        prob = probabilities[i]
        true_label = true_labels[i].item()
        predicted_label = predicted_labels[i]
        threshold_used = threshold_array[i]

        if predicted_label == 1 and true_label == 1:
            status_tag = "TP"
            correct_predictions += 1 # True Positive
        elif predicted_label == 0 and true_label == 0:
            status_tag = "TN"
            correct_predictions += 1 # True Negative
        elif predicted_label == 1 and true_label == 0:
            status_tag = "FP" # False Positive
        else:
            status_tag = "FN" # False Negative

        prediction_word = "Yes" if predicted_label == 1 else "No"
        pred_text += f" {status_tag}: {disease:<20} ({prob:.3f} > {threshold_used:.2f}) [{prediction_word}]\n"

    # --- Counter
    accuracy_text = f"\n{correct_predictions} / {total_classes}\n"
    # -----------------------------------
    
    text_output = f"{true_text}{accuracy_text}\n{pred_text}"

    ax[1].text(0.05, 0.95, text_output,
              transform=ax[1].transAxes,
              fontsize=20,
              verticalalignment='top',
              family='monospace')
    ax[1].set_title("Prediction vs Ground Truth (Per-Class Thresholds)", fontsize=14)
    ax[1].axis('off')

    plt.tight_layout()
    plt.show()


# -------------------------------------------------------------------
# Main function
# -------------------------------------------------------------------
def main():
    # Load optimal thresholds
    try:
        with open(PATH_THRESHOLDS, 'r') as f:
            optimal_thresholds = json.load(f)
        print(f"Optimal thresholds loaded from: {PATH_THRESHOLDS}")
    except FileNotFoundError:
        print(f"Error: Threshold file '{PATH_THRESHOLDS}' not found. Run tune_thresholds.py first.")
        sys.exit(1)
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in threshold file.")
        sys.exit(1)

    # Find a positive sample for visualization
    found_positive_case = False
    max_attempts = 100

    try:
        _, _, _, _, DATASET_SIZE = get_single_data_point(0)
    except Exception as e:
        print(f"Error initializing dataset: {e}")
        sys.exit(1)

    print(f"Searching for a sample with at least one positive label (out of {DATASET_SIZE})...")

    INDEX_TO_CHECK = -1
    for attempt in range(max_attempts):
        INDEX_TO_CHECK = random.randint(0, DATASET_SIZE - 1)
        try:
            _, true_labels, _, _, _ = get_single_data_point(INDEX_TO_CHECK)
        except Exception as e:
            print(f"Error loading data at index {INDEX_TO_CHECK}: {e}")
            continue

        if torch.sum(true_labels) > 0:
            found_positive_case = True
            break

    if found_positive_case:
        print(f"Found positive sample: Index = {INDEX_TO_CHECK}")
        visualize_prediction(INDEX_TO_CHECK, optimal_thresholds)
    else:
        print(f"Could not find a positive sample after {max_attempts} attempts.")


if __name__ == '__main__':
    main()