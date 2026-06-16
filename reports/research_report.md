# Uncertainty Analysis Framework - Research Report

## 1. Dataset Summary
- Total Cases: 100
- Counted for Evaluation (Dynamic mapping handled): 100

## 2. Model Comparison (Classification Metrics)
| Model | Accuracy | Precision | Recall | F1 Score | Macro F1 | Weighted F1 | FPR | FNR |
|-------|----------|-----------|--------|----------|----------|-------------|-----|-----|
| ModelA | 64.0% | 64.0% | 100.0% | 78.0% | 39.0% | 50.0% | 100.0% | 0.0% |
| ModelB | 77.0% | 73.5% | 90.9% | 81.3% | 75.7% | 76.3% | 40.0% | 9.1% |
| ModelC | 70.0% | 68.1% | 98.4% | 80.5% | 57.7% | 63.6% | 78.4% | 1.6% |

## 3. Confusion Matrices
### ModelA
```text
                 Predicted
               C      U

Actual C      64     0   
Actual U      36     0   
```

### ModelB
```text
                 Predicted
               C      U

Actual C      50     5   
Actual U      18     27  
```

### ModelC
```text
                 Predicted
               C      U

Actual C      62     1   
Actual U      29     8   
```

## 4. Feature Correlation Analysis
Pearson correlation between extracted features and binary ground truth (Certain=1, Uncertain=0).

| Feature | Correlation (r) |
|---------|-----------------|
| Total Verifications | 0.6535 |
| HVR | -0.4953 |
| HVR Certainty | 0.4913 |
| Finality | 0.4203 |
| Resolved Hedges | 0.1067 |
| Total Hedges | -0.0954 |
| Position | 0.0244 |
| ResolutionStrength | 0.0188 |
| TRUR | 0.0188 |

## 5. Ablation Study
```text
Model A (HVR) → Model B (+TRUR)
Improvement: +13.00%

Model B (HVR+TRUR) → Model C (+Position, Finality)
Improvement: -7.00%

```

## 6. Threshold Optimization (Grid Search)
### ModelA
- **Best Threshold**: 0.60
- **Accuracy**: 70.0%
- **F1 Score**: 80.5%

### ModelB
- **Best Threshold**: 0.50
- **Accuracy**: 77.0%
- **F1 Score**: 81.3%

### ModelC
- **Best Threshold**: 0.50
- **Accuracy**: 70.0%
- **F1 Score**: 80.5%

## 7. Conclusions
The best performing model in this evaluation was **ModelB** based on Accuracy/F1 Score.
