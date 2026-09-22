"""
Abstract base classes for image analyzers.
Enables pluggable computer vision / deep learning models (CLIP, YOLO, Vision LLMs).
"""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
from app.models.schemas import ReferenceProfile, ParticipantProfile


class BaseReferenceAnalyzer(ABC):
    """
    Modular interface for analyzing the single original reference image.
    Generates a structured ReferenceProfile detailing subjects, importance, colors,
    composition, spatial relationships, and fine details.
    """

    @abstractmethod
    def analyze(self, image_path: Path, image_id: Optional[str] = None) -> ReferenceProfile:
        """
        Analyze reference image and generate structured ReferenceProfile.

        :param image_path: Path to the validated reference image on disk.
        :param image_id: Optional unique identifier for the competition session.
        :return: Populated ReferenceProfile.
        """
        pass


class BaseParticipantAnalyzer(ABC):
    """
    Modular interface for analyzing a participant's recreation image.
    Extracts features aligned with the reference profile's dimensions.
    """

    @abstractmethod
    def analyze(
        self,
        image_path: Path,
        participant_id: str,
        reference_profile: Optional[ReferenceProfile] = None
    ) -> ParticipantProfile:
        """
        Analyze participant image and extract features.

        :param image_path: Path to the validated participant image on disk.
        :param participant_id: Distinct participant identifier (e.g. 'PARTICIPANT-01').
        :param reference_profile: The reference profile for context-guided analysis.
        :return: Populated ParticipantProfile.
        """
        pass
