import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def resolve_ground_truth(expected_label: str, certainty_level: str, prediction: str) -> str:
    """
    Map the 4-class ground truth into binary ('Certain', 'Uncertain') dynamically.
    
    Rules based on user feedback:
    - Expected 'Certain' or 'Resolved Uncertainty' -> 'Certain'
    - Expected 'Unresolved Uncertainty' -> 'Uncertain'
    - Expected 'Partially Resolved':
        - If CertaintyLevel == 'Low' -> 'Uncertain'
        - If CertaintyLevel == 'Medium' -> Accept whatever the model predicted
    """
    if expected_label in ("Certain", "Resolved Uncertainty"):
        return "Certain"
    if expected_label == "Unresolved Uncertainty":
        return "Uncertain"
        
    # Handle Partially Resolved
    if expected_label == "Partially Resolved":
        if certainty_level == "Low":
            return "Uncertain"
        elif certainty_level == "Medium":
            # Dynamic mapping: treat the prediction as correct
            return prediction
            
    # Fallback (should not happen with standard dataset)
    return "Uncertain"

def compute_metrics(y_true, y_pred):
    """
    Compute binary classification metrics using scikit-learn.
    We map 'Certain' -> 1 and 'Uncertain' -> 0 for sklearn.
    """
    # Map to binary integers
    y_true_bin = [1 if y == "Certain" else 0 for y in y_true]
    y_pred_bin = [1 if y == "Certain" else 0 for y in y_pred]
    
    acc = accuracy_score(y_true_bin, y_pred_bin)
    
    # zero_division=0 handles cases where a model predicts 0 positives
    prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    
    macro_f1 = f1_score(y_true_bin, y_pred_bin, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true_bin, y_pred_bin, average='weighted', zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    return {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1,
        "FPR": fpr,
        "FNR": fnr
    }
