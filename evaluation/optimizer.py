from evaluation.metrics import compute_metrics

def optimize_thresholds(model, feature_sets, y_true, thresholds):
    """
    Grid search for the best threshold for a specific model based on Accuracy (or F1).
    """
    best_thresh = 0.50
    best_acc = -1.0
    best_f1 = -1.0
    
    results = []
    
    for t in thresholds:
        y_pred = [model.predict(fs, threshold=t) for fs in feature_sets]
        metrics = compute_metrics(y_true, y_pred)
        
        acc = metrics["Accuracy"]
        f1 = metrics["F1 Score"]
        
        results.append((t, acc, f1))
        
        if acc > best_acc or (acc == best_acc and f1 > best_f1):
            best_acc = acc
            best_f1 = f1
            best_thresh = t
            
    return {
        "model_name": model.name,
        "best_threshold": best_thresh,
        "best_accuracy": best_acc,
        "best_f1": best_f1,
        "grid": results
    }
