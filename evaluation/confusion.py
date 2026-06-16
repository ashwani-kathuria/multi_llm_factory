from sklearn.metrics import confusion_matrix
import pandas as pd

def generate_confusion_matrix(y_true, y_pred):
    """
    Generate a formatted confusion matrix string.
    y_true and y_pred contain strings 'Certain' or 'Uncertain'.
    """
    y_true_bin = [1 if y == "Certain" else 0 for y in y_true]
    y_pred_bin = [1 if y == "Certain" else 0 for y in y_pred]
    
    # labels=[0, 1] means:
    # row 0 = Actual Uncertain, row 1 = Actual Certain
    # col 0 = Pred Uncertain, col 1 = Pred Certain
    cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
    
    tn, fp = cm[0]
    fn, tp = cm[1]
    
    # We want format:
    #                  Predicted
    #                C      U
    # Actual C      TP     FN
    # Actual U      FP     TN
    
    matrix_str = (
        "                 Predicted\n"
        "               C      U\n\n"
        f"Actual C      {tp:<4}   {fn:<4}\n"
        f"Actual U      {fp:<4}   {tn:<4}\n"
    )
    return matrix_str
