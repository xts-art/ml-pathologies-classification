import torch
import os
import sys

# --- 1. SPECIFY THE CORRECT FILE PATH ---
checkpoint_path = r'F:\gdtnetpp_project\training_checkpoint.pth' 
# ------------------------------------------

try:
    print(f"Loading file: {checkpoint_path}")
    # Load the checkpoint file, mapping everything to the CPU
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    print("File successfully loaded. Object type:", type(checkpoint))

except Exception as e:
    print(f"An error occurred while loading the PyTorch file: {e}")
    sys.exit()

# ----------------------------------------------------
# --- 2. FULL ANALYSIS OF THE MODEL_STATE_DICT STRUCTURE ---
# ----------------------------------------------------

print("\n" + "="*80)
print("             FULL ANALYSIS OF MODEL_STATE_DICT (All 727 Layers)")
print("="*80)

# Find the key that contains the model weights
state_dict = None
sd_key = None
if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
    sd_key = 'model_state_dict'
elif 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
    sd_key = 'state_dict'

if state_dict and isinstance(state_dict, dict):
    
    keys_list = list(state_dict.keys())
    
    print(f"Found **{len(keys_list)}** layers/tensors in key '{sd_key}'.")
    print("-" * 80)

    # Output information about ALL keys
    for i, name in enumerate(keys_list):
        param = state_dict[name]
        
        # If it's a PyTorch tensor (weights), output its size and data type
        if hasattr(param, 'size') and isinstance(param, torch.Tensor):
            size_info = f"Size: {tuple(param.size())}, Data type: {param.dtype}"
        else:
            # If it's not a tensor
            size_info = f"Type: {type(param).__name__}"
            
        # Formatted output
        print(f"[{i+1}/{len(keys_list)}] {name:<60}: {size_info}")
        
    print("-" * 80)
    print("state_dict output complete.")

# ----------------------------------------------------

# 3. Output metadata (for completeness)
print("\n" + "="*80)
print("                       METADATA SUMMARY")
print("="*80)

print(f"   - Last Epoch Number: **{checkpoint.get('epoch', 'N/A')}**")
print(f"   - Val Loss: **{checkpoint.get('val_loss', 'N/A')}**")
print(f"   - Best Val Loss: **{checkpoint.get('best_val_loss', 'N/A')}**")
print(f"   - Best Mean AUC: **{checkpoint.get('best_mean_auc', 'N/A')}**")
print(f"   - Optimizer State: {'PRESENT' if 'optimizer_state_dict' in checkpoint else 'ABSENT'}")
print("="*80)