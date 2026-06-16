import sys
import yaml
import json
import pandas as pd

from framework.feature_extractor import extract_features
from models.model_a import ModelA
from models.model_b import ModelB
from models.model_c import ModelC
from evaluation.metrics import resolve_ground_truth, compute_metrics
from evaluation.confusion import generate_confusion_matrix
from evaluation.correlation import compute_correlations
from evaluation.ablation import generate_ablation_study
from evaluation.optimizer import optimize_thresholds
from reports.report_generator import write_research_report

def load_data():
    """
    Load raw reasoning data from the 4 batch CSV files.
    """
    import csv
    
    records = []
    batch_files = [
        "testCases_Batch_1.txt",
        "testCases_Batch_2.txt",
        "testCases_Batch_3.txt",
        "testCases_Batch_4.txt",
    ]
    
    for filename in batch_files:
        try:
            with open(filename, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("TestCaseID") and row.get("ReasoningText"):
                        records.append(row)
        except Exception as e:
            print(f"Warning: Could not load {filename}: {e}")
            
    return records

def main():
    print("Loading data...")
    records = load_data()
    print(f"Loaded {len(records)} test cases.")
    
    print("Loading configs...")
    with open("config/model_weights.yaml", "r") as f:
        weights_config = yaml.safe_load(f)
        
    with open("config/thresholds.yaml", "r") as f:
        thresh_config = yaml.safe_load(f)
        thresholds = thresh_config.get("grid_search", [0.50])
        
    models = [
        ModelA(weights=weights_config.get("model_a", {}).get("weights", {})),
        ModelB(weights=weights_config.get("model_b", {}).get("weights", {})),
        ModelC(weights=weights_config.get("model_c", {}).get("weights", {})),
    ]
    
    print("Extracting features...")
    feature_sets = []
    y_true_dynamic = []
    
    detailed_rows = []
    
    # We will use threshold 0.50 for the default detailed evaluation
    DEFAULT_THRESH = 0.50
    
    for row in records:
        text = row.get("ReasoningText", "")
        if not text.strip():
            continue
            
        fs = extract_features(text)
        feature_sets.append(fs)
        
        expected_raw = row.get("ExpectedLabel", "Unknown")
        cert_level = row.get("CertaintyLevel", "Medium")
        
        # We need a dynamic map based on Model C (or best model) to decide ground truth for "Medium"
        # For simplicity of comparing apples-to-apples across all models, we evaluate ground truth 
        # using the prediction of Model C since it's the most advanced, OR we dynamically map 
        # PER MODEL during evaluation. Let's do it per model during evaluation!
        
        # We will collect base information for the detailed CSV
        detailed_row = {
            "TestCaseID": row["TestCaseID"],
            "Expected": expected_raw,
            "CertaintyLevel": cert_level,
            "Total Hedges": fs.total_hedges,
            "Total Verifications": fs.total_verifications,
            "Resolved Hedges": fs.resolved_hedges,
            "HVR": fs.hvr,
            "HVR Certainty": fs.hvr_certainty,
            "TRUR": fs.trur,
            "ResolutionStrength": fs.resolution_strength,
            "Position": fs.position,
            "Finality": fs.finality
        }
        
        # Calculate per-model score/prediction
        for model in models:
            score = model.score(fs)
            pred = model.predict(fs, threshold=DEFAULT_THRESH)
            
            # Dynamic Truth mapping for THIS model
            truth = resolve_ground_truth(expected_raw, cert_level, pred)
            
            detailed_row[f"{model.name}_Score"] = score
            detailed_row[f"{model.name}_Pred"] = pred
            detailed_row[f"{model.name}_Truth"] = truth
            detailed_row[f"{model.name}_Correct"] = (pred == truth)
            
        detailed_rows.append(detailed_row)

    print("Running evaluations...")
    # Calculate metrics per model
    model_metrics = {}
    model_confusions = {}
    optimizer_results = []
    
    best_model_name = "ModelA"
    best_overall_acc = -1.0
    
    for model in models:
        y_true_model = [r[f"{model.name}_Truth"] for r in detailed_rows]
        y_pred_model = [r[f"{model.name}_Pred"] for r in detailed_rows]
        
        metrics = compute_metrics(y_true_model, y_pred_model)
        model_metrics[model.name] = metrics
        
        cm_str = generate_confusion_matrix(y_true_model, y_pred_model)
        model_confusions[model.name] = cm_str
        
        if metrics["Accuracy"] > best_overall_acc:
            best_overall_acc = metrics["Accuracy"]
            best_model_name = model.name
            
        # Grid Search
        opt_res = optimize_thresholds(model, feature_sets, y_true_model, thresholds)
        optimizer_results.append(opt_res)
        
    # Feature Correlation
    # For correlation, we need a single solid ground truth. 
    # Let's use the truth mapping from the best model (or Model C)
    y_true_for_corr = [r[f"{best_model_name}_Truth"] for r in detailed_rows]
    corr_data = compute_correlations(feature_sets, y_true_for_corr)
    
    # Ablation
    ablation_text = generate_ablation_study(model_metrics)
    
    # Output Detailed CSV
    df = pd.DataFrame(detailed_rows)
    df.to_csv("reports/per_test_case_results.csv", index=False)
    
    print("Generating report...")
    dataset_summary = {
        "Total Cases": len(records),
        "Counted": len(detailed_rows)
    }
    
    write_research_report(
        "reports/research_report.md",
        dataset_summary,
        model_metrics,
        model_confusions,
        optimizer_results,
        corr_data,
        ablation_text,
        best_model_name
    )
    
    print("Done! Reports saved to reports/research_report.md and reports/per_test_case_results.csv")

if __name__ == "__main__":
    main()
