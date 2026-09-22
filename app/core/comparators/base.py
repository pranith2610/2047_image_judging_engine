"""
Abstract base class for feature comparison.
Allows plugging in CLIP cosine similarity or multi-modal vector search.
"""
from abc import ABC, abstractmethod
from app.models.schemas import (
    ReferenceProfile,
    ParticipantProfile,
    FeatureComparisonMetrics,
)


class BaseFeatureComparator(ABC):
    """Modular interface for comparing participant recreation against reference profile."""

    @abstractmethod
    def compare(
        self, reference: ReferenceProfile, participant: ParticipantProfile
    ) -> FeatureComparisonMetrics:
        """
        Compare extracted features and compute normalized metrics [0.0, 1.0].

        :param reference: Structured reference profile with importance weights.
        :param participant: Extracted participant profile.
        :return: Multi-dimensional comparison metrics.
        """
        pass
