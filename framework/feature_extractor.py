import sys
import os

# Add parent directory to path so we can import the original proximity_metric
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from proximity_metric import calculate_proximity_metrics
from framework.datatypes import FeatureSet

def extract_features(reasoning_text: str) -> FeatureSet:
    """
    Runs the existing proximity_metric pipeline to detect hedges, verifications,
    and resolutions, then maps the raw outputs into a normalized FeatureSet
    ready for any multi-model evaluation.
    """
    # The existing code does steps 1-8 internally and returns a dict
    # of raw and calculated metrics. We just need to extract the raw metrics
    # and map them to our FeatureSet.
    raw_metrics = calculate_proximity_metrics(reasoning_text)
    
    total_hedges = raw_metrics.get("total_hedges", 0)
    total_verifs = raw_metrics.get("total_verifications", 0)
    resolved = raw_metrics.get("resolved_hedges", 0)
    
    # 1. HVR (Hedge-to-Verify Ratio)
    # The formula defined in prompt: HVR = TotalHedges / (TotalVerificationWords + 1)
    # We will use this exact formula as per requirement.
    hvr = total_hedges / (total_verifs + 1)
    
    # 2. HVR_Certainty = 1 / (1 + HVR)
    hvr_certainty = 1.0 / (1.0 + hvr)
    
    # 3. TRUR = ResolvedHedges / TotalHedges (or 0 if no hedges)
    trur = (resolved / total_hedges) if total_hedges > 0 else 0.0
    
    # 4. Resolution Strength
    # Defined as mathematically identical to TRUR for this iteration
    resolution_strength = trur
    
    # 5. Position
    # The existing codebase calculates 'late_unresolved_ratio'. 
    # Position in the prompt represents "fraction of unresolved hedges in the first half (early = Certain)"
    # We map this as (1.0 - late_unresolved_ratio). 
    # If no hedges, it's a neutral/perfect 1.0 (since there are no late unresolved hedges to penalize)
    late_unresolved = raw_metrics.get("late_unresolved_ratio", 0.0)
    position = 1.0 - late_unresolved
    
    # 6. Finality
    finality = raw_metrics.get("conclusion_finality", 0.0)
    
    return FeatureSet(
        total_hedges=total_hedges,
        total_verifications=total_verifs,
        resolved_hedges=resolved,
        hvr=hvr,
        hvr_certainty=hvr_certainty,
        trur=trur,
        resolution_strength=resolution_strength,
        position=position,
        finality=finality
    )
