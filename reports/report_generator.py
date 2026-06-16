import pandas as pd
from typing import List, Dict

def write_research_report(
    filepath: str,
    dataset_summary: dict,
    model_metrics: dict,
    model_confusions: dict,
    optimizer_results: list,
    correlation_data: pd.Series,
    ablation_text: str,
    best_model_name: str,
):
    """
    Generate the final Markdown research report.
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("# Uncertainty Analysis Framework - Research Report\n\n")
        
        # 1. Dataset Summary
        f.write("## 1. Dataset Summary\n")
        f.write(f"- Total Cases: {dataset_summary['Total Cases']}\n")
        f.write(f"- Counted for Evaluation (Dynamic mapping handled): {dataset_summary['Counted']}\n\n")
        
        # 2. Model Comparison Table
        f.write("## 2. Model Comparison (Classification Metrics)\n")
        f.write("| Model | Accuracy | Precision | Recall | F1 Score | Macro F1 | Weighted F1 | FPR | FNR |\n")
        f.write("|-------|----------|-----------|--------|----------|----------|-------------|-----|-----|\n")
        for m_name, metrics in model_metrics.items():
            f.write(f"| {m_name} "
                    f"| {metrics['Accuracy']*100:.1f}% "
                    f"| {metrics['Precision']*100:.1f}% "
                    f"| {metrics['Recall']*100:.1f}% "
                    f"| {metrics['F1 Score']*100:.1f}% "
                    f"| {metrics['Macro F1']*100:.1f}% "
                    f"| {metrics['Weighted F1']*100:.1f}% "
                    f"| {metrics['FPR']*100:.1f}% "
                    f"| {metrics['FNR']*100:.1f}% |\n")
        f.write("\n")
        
        # 3. Confusion Matrices
        f.write("## 3. Confusion Matrices\n")
        for m_name, cm_str in model_confusions.items():
            f.write(f"### {m_name}\n```text\n{cm_str}```\n\n")
            
        # 4. Feature Correlation Analysis
        f.write("## 4. Feature Correlation Analysis\n")
        f.write("Pearson correlation between extracted features and binary ground truth (Certain=1, Uncertain=0).\n\n")
        f.write("| Feature | Correlation (r) |\n")
        f.write("|---------|-----------------|\n")
        for feature, r_val in correlation_data.items():
            f.write(f"| {feature} | {r_val:.4f} |\n")
        f.write("\n")
        
        # 5. Ablation Study
        f.write("## 5. Ablation Study\n")
        f.write("```text\n")
        f.write(ablation_text)
        f.write("\n```\n\n")
        
        # 6. Threshold Optimization
        f.write("## 6. Threshold Optimization (Grid Search)\n")
        for opt in optimizer_results:
            f.write(f"### {opt['model_name']}\n")
            f.write(f"- **Best Threshold**: {opt['best_threshold']:.2f}\n")
            f.write(f"- **Accuracy**: {opt['best_accuracy']*100:.1f}%\n")
            f.write(f"- **F1 Score**: {opt['best_f1']*100:.1f}%\n\n")
            
        # 7. Conclusions
        f.write("## 7. Conclusions\n")
        f.write(f"The best performing model in this evaluation was **{best_model_name}** based on Accuracy/F1 Score.\n")
