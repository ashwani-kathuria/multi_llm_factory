def generate_ablation_study(metrics_results: dict) -> str:
    """
    Compare models A -> B -> C to show the incremental improvement
    of adding features.
    metrics_results is a dict of Model Name -> metrics dict
    e.g., {"ModelA": {"Accuracy": 0.8, ...}, ...}
    """
    lines = []
    
    model_a_acc = metrics_results.get("ModelA", {}).get("Accuracy", 0)
    model_b_acc = metrics_results.get("ModelB", {}).get("Accuracy", 0)
    model_c_acc = metrics_results.get("ModelC", {}).get("Accuracy", 0)
    
    # A -> B (Adding TRUR)
    if "ModelA" in metrics_results and "ModelB" in metrics_results:
        diff_b_a = (model_b_acc - model_a_acc) * 100
        sign = "+" if diff_b_a >= 0 else ""
        lines.append(f"Model A (HVR) → Model B (+TRUR)")
        lines.append(f"Improvement: {sign}{diff_b_a:.2f}%\n")
        
    # B -> C (Adding Position and Finality)
    if "ModelB" in metrics_results and "ModelC" in metrics_results:
        diff_c_b = (model_c_acc - model_b_acc) * 100
        sign = "+" if diff_c_b >= 0 else ""
        lines.append(f"Model B (HVR+TRUR) → Model C (+Position, Finality)")
        lines.append(f"Improvement: {sign}{diff_c_b:.2f}%\n")
        
    return "\n".join(lines)
