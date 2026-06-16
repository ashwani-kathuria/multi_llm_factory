from framework.datatypes import FeatureSet
from models.base_model import BaseModel

class ModelC(BaseModel):
    """
    Model C: HVR + Position + TRUR + Finality
    """
    def __init__(self, weights: dict):
        super().__init__("ModelC", weights)
        self.w_hvr = self.weights.get('hvr', 0.45)
        self.w_pos = self.weights.get('position', 0.25)
        self.w_trur = self.weights.get('trur', 0.15)
        self.w_fin = self.weights.get('finality', 0.15)

    def score(self, features: FeatureSet) -> float:
        # Finality ranges from -1 to 1, we normalize it to 0-1
        normalized_finality = (features.finality + 1.0) / 2.0
        
        return (
            self.w_hvr * features.hvr_certainty +
            self.w_pos * features.position +
            self.w_trur * features.trur +
            self.w_fin * normalized_finality
        )

    def explain_score(self, features: FeatureSet) -> str:
        normalized_finality = (features.finality + 1.0) / 2.0
        s_hvr = self.w_hvr * features.hvr_certainty
        s_pos = self.w_pos * features.position
        s_trur = self.w_trur * features.trur
        s_fin = self.w_fin * normalized_finality
        
        return (
            f"Score = {s_hvr:.4f} ({self.w_hvr}*HVR_C) + "
            f"{s_pos:.4f} ({self.w_pos}*Pos) + "
            f"{s_trur:.4f} ({self.w_trur}*TRUR) + "
            f"{s_fin:.4f} ({self.w_fin}*Fin_Norm) = {self.score(features):.4f}"
        )
