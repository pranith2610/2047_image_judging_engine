"""
Enumerations for THE 2047 judging engine.
"""
from enum import Enum


class ImportanceLevel(str, Enum):
    """Relative importance weights for visual elements in the reference image."""
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

    @property
    def weight(self) -> float:
        """Numerical weight for importance (HIGH=1.0, MEDIUM=0.6, LOW=0.25)."""
        if self == ImportanceLevel.HIGH:
            return 1.0
        elif self == ImportanceLevel.MEDIUM:
            return 0.6
        elif self == ImportanceLevel.LOW:
            return 0.25
        return 1.0

    @property
    def weight_multiplier(self) -> float:
        """Numerical weight multiplier for importance."""
        if self == ImportanceLevel.HIGH:
            return 1.5
        elif self == ImportanceLevel.MEDIUM:
            return 1.0
        elif self == ImportanceLevel.LOW:
            return 0.5
        return 1.0


class ObjectMatchStatus(str, Enum):
    """Object detection and alignment classification."""
    PRESENT = "PRESENT"
    PARTIALLY_PRESENT = "PARTIALLY PRESENT"
    MISSING = "MISSING"
    INCORRECT = "INCORRECT"


class ScoreCategory(str, Enum):
    """The 5 official scoring categories totaling 100 points."""
    SEMANTIC_SIMILARITY = "semantic_similarity"       # Max 30 points
    OBJECT_ACCURACY = "object_accuracy"               # Max 25 points
    COMPOSITION_SPATIAL = "composition_spatial"       # Max 20 points
    COLOR_LIGHTING = "color_lighting"                 # Max 15 points
    FINE_DETAILS = "fine_details"                     # Max 10 points


class SpatialQuadrant(str, Enum):
    """Spatial composition layout zones."""
    TOP_LEFT = "top_left"
    TOP_CENTER = "top_center"
    TOP_RIGHT = "top_right"
    CENTER_LEFT = "center_left"
    CENTER = "center"
    CENTER_RIGHT = "center_right"
    BOTTOM_LEFT = "bottom_left"
    BOTTOM_CENTER = "bottom_center"
    BOTTOM_RIGHT = "bottom_right"


class AnalysisStatus(str, Enum):
    """Status of image analysis processing."""
    PENDING = "PENDING"
    ANALYZING = "ANALYZING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class SceneType(str, Enum):
    """Indoor or outdoor scene categorization."""
    INDOOR = "indoor"
    OUTDOOR = "outdoor"
    STUDIO_ABSTRACT = "studio_abstract"
