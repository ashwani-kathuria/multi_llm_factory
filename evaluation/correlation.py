import pandas as pd

def compute_correlations(feature_sets, y_true):
    """
    Compute Pearson correlation between each feature and the binary ground truth.
    y_true: list of 'Certain' or 'Uncertain'
    feature_sets: list of FeatureSet objects
    """
    y_true_bin = [1 if y == "Certain" else 0 for y in y_true]
    
    # Convert features to DataFrame
    df = pd.DataFrame([fs.to_dict() for fs in feature_sets])
    df['GroundTruth'] = y_true_bin
    
    # Calculate correlations
    corr = df.corr()['GroundTruth'].drop('GroundTruth')
    
    # Sort by absolute correlation strength
    corr_sorted = corr.reindex(corr.abs().sort_values(ascending=False).index)
    
    return corr_sorted
