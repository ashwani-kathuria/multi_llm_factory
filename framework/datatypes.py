from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class FeatureSet:
    """Represents the raw mathematical features extracted from a reasoning trace."""
    total_hedges: int
    total_verifications: int
    resolved_hedges: int

    hvr: float
    hvr_certainty: float

    trur: float
    
    # Model D explicitly required ResolutionStrength to be mathematically equal
    # to TRUR for this iteration, but named separately to support future logic divergence.
    resolution_strength: float

    position: float
    finality: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert features to dictionary for easy CSV/Pandas export."""
        return {
            "Total Hedges": self.total_hedges,
            "Total Verifications": self.total_verifications,
            "Resolved Hedges": self.resolved_hedges,
            "HVR": round(self.hvr, 4),
            "HVR Certainty": round(self.hvr_certainty, 4),
            "TRUR": round(self.trur, 4),
            "ResolutionStrength": round(self.resolution_strength, 4),
            "Position": round(self.position, 4),
            "Finality": round(self.finality, 4),
        }
