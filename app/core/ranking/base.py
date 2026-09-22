"""
Abstract base class for Ranking Engine.
Orders participants and handles deterministic tie-breaking.
"""
from abc import ABC, abstractmethod
from typing import List
from app.models.schemas import EvaluationResult, RankedParticipant


class BaseRankingEngine(ABC):
    """
    Modular interface for producing competition rankings.
    """

    @abstractmethod
    def rank(self, evaluations: List[EvaluationResult]) -> List[RankedParticipant]:
        """
        Rank participants based on total and category-wise performance.

        :param evaluations: List of evaluated participants.
        :return: Ordered list of RankedParticipant with assigned rank and tiers.
        """
        pass
