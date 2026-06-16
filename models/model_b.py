from framework.datatypes import FeatureSet
from models.base_model import BaseModel

class ModelB(BaseModel):
    """
    Model B: HVR + TRUR
    """
    def __init__(self, weights: dict):
        super().__init__("ModelB", weights)
        self.w_hvr = self.weights.get('hvr', 0.70)
        self.w_trur = self.weights.get('trur', 0.30)

    def score(self, features: FeatureSet) -> float:
        return (
            self.w_hvr * features.hvr_certainty +
            self.w_trur * features.trur
        )

    def explain_score(self, features: FeatureSet) -> str:
        s_hvr = self.w_hvr * features.hvr_certainty
        s_trur = self.w_trur * features.trur
        return (
            f"Score = {s_hvr:.4f} ({self.w_hvr} * HVR_C) + "
            f"{s_trur:.4f} ({self.w_trur} * TRUR) = {self.score(features):.4f}"
        )
