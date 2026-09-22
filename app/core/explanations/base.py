"""
Abstract base class for Explanation Engine.
Generates natural language rationale for awarded competition scores.
"""
from abc import ABC, abstractmethod
from typing import Tuple, Dict
from app.models.schemas import (
    ReferenceProfile,
    ParticipantProfile,
    CategoryScores,
    FeatureComparisonMetrics,
    JudgingReport,
)


class BaseExplanationEngine(ABC):
    """
    Modular interface for explaining scores to participants and judges.
    Can be powered by rule-based heuristics or local/remote LLM reasoning.
    """

    @abstractmethod
    def generate_explanation(
        self,
        reference: ReferenceProfile,
        participant: ParticipantProfile,
        metrics: FeatureComparisonMetrics,
        scores: CategoryScores,
    ) -> Tuple[str, Dict[str, str]]:
        """
        Generate human-readable explanation and category-by-category justifications.

        :param reference: Reference profile.
        :param participant: Participant profile.
        :param metrics: Normalized comparison metrics.
        :param scores: Official category scores.
        :return: Tuple of (overall_summary_explanation, dict_of_category_reasons).
        """
        pass

    @abstractmethod
    def generate_judging_report(
        self,
        reference: ReferenceProfile,
        participant: ParticipantProfile,
        metrics: FeatureComparisonMetrics,
        scores: CategoryScores,
    ) -> JudgingReport:
        """
        Generate complete structured judging report for a participant.

        :param reference: Reference profile.
        :param participant: Participant profile.
        :param metrics: Normalized comparison metrics.
        :param scores: Official category scores.
        :return: JudgingReport data model.
        """
        pass
