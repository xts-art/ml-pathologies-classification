# src/focal_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification.
    It modifies BCEWithLogitsLoss to focus more on hard-to-classify examples.
    """
    
    def __init__(self, gamma=2.0, alpha=None, reduction='mean', pos_weight=None):
        """
        :param gamma: Focusing parameter. Higher values focus more on hard examples (typical 0.5 - 5)
        :param alpha: Weighting factor for positive/negative balance. Ignored if pos_weight is provided.
        :param pos_weight: Class weights for positive labels (used by BCEWithLogitsLoss)
        :param reduction: 'mean', 'sum', or 'none'
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):
        """
        Compute Focal Loss between inputs (logits) and targets.
        """
        # --- 1. Compute BCE loss (element-wise) ---
        if self.pos_weight is not None:
            # Multi-label BCE with pos_weight
            bce_loss = F.binary_cross_entropy_with_logits(
                inputs, targets, pos_weight=self.pos_weight, reduction='none'
            )
        else:
            # Standard BCE
            bce_loss = F.binary_cross_entropy_with_logits(
                inputs, targets, reduction='none'
            )
        
        # --- 2. Convert logits to probabilities ---
        probs = torch.sigmoid(inputs)
        
        # --- 3. Compute Pt (probability of the true class) ---
        # Pt = p if target=1, Pt = 1-p if target=0
        pt = targets * probs + (1 - targets) * (1 - probs)
        
        # --- 4. Compute the focal term ---
        focal_term = (1 - pt) ** self.gamma
        
        # --- 5. Alpha weighting (ignored here if pos_weight is used) ---
        # In this implementation, pos_weight already handles class imbalance.
        
        # --- 6. Final Focal Loss ---
        loss = focal_term * bce_loss
        
        # --- 7. Apply reduction ---
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
