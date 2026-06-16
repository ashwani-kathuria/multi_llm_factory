from framework.datatypes import FeatureSet
from models.base_model import BaseModel

class ModelA(BaseModel):
    """
    Model A: HVR Only
    Score = HVR_Certainty
    """
    def __init__(self, weights=None):
        super().__init__("ModelA", weights)

    def score(self, features: FeatureSet) -> float:
        return features.hvr_certainty

    def explain_score(self, features: FeatureSet) -> str:
        return f"Score = {features.hvr_certainty:.4f} (HVR_Certainty)"
