from abc import ABC, abstractmethod
from typing import Dict, Any
from framework.datatypes import FeatureSet

class BaseModel(ABC):
    """
    Abstract base class for all uncertainty scoring models.
    """
    def __init__(self, name: str, weights: Dict[str, float] = None):
        self.name = name
        self.weights = weights or {}
        
    @abstractmethod
    def score(self, features: FeatureSet) -> float:
        """Calculate the uncertainty score between 0.0 (Uncertain) and 1.0 (Certain)."""
        pass
        
    def predict(self, features: FeatureSet, threshold: float = 0.50) -> str:
        """Predict 'Certain' or 'Uncertain' based on the score and threshold."""
        score_val = self.score(features)
        return "Certain" if score_val >= threshold else "Uncertain"
        
    def explain_score(self, features: FeatureSet) -> str:
        """Return a string explaining the mathematical calculation for transparency."""
        pass
