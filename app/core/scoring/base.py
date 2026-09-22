"""
Abstract base class for the Scoring Engine.
Calculates 100-point category-wise competition scores.
"""
from abc import ABC, abstractmethod
from app.models.schemas import (
    FeatureComparisonMetrics,
    ReferenceProfile,
    CategoryScores,
)


class BaseScoringEngine(ABC):
    """
    Modular interface for calculating official competition scores.
    Strictly constrained to 100 total points across the 5 categories.
    """

    @abstractmethod
    def calculate_scores(
        self, metrics: FeatureComparisonMetrics, reference: ReferenceProfile
    ) -> CategoryScores:
        """
        Compute category-wise scores and total score out of 100.

        :param metrics: Normalized comparison metrics [0.0, 1.0].
        :param reference: Reference profile containing element importance specifications.
        :return: CategoryScores instance summing to max 100.0.
        """
        pass
