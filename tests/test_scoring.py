"""
Unit tests for the weighted scoring engine.
"""
import pytest
from app.models.schemas import (
    FeatureComparisonMetrics,
    ReferenceProfile,
    VisualElement,
    ColorProfile,
    CompositionProfile,
    DetailProfile,
)
from app.models.enums import ImportanceLevel
from app.core.scoring.engine import WeightedScoringEngine
from app.config import CATEGORY_WEIGHTS, TOTAL_POINTS


@pytest.fixture
def sample_reference_profile() -> ReferenceProfile:
    return ReferenceProfile(
        image_id="REF-TEST",
        filename="ref.png",
        dimensions=(800, 600),
        main_subjects=[
            VisualElement(name="Person", importance=ImportanceLevel.HIGH),
            VisualElement(name="Red Umbrella", importance=ImportanceLevel.HIGH),
        ],
        secondary_subjects=[
            VisualElement(name="Rain", importance=ImportanceLevel.HIGH),
            VisualElement(name="Street", importance=ImportanceLevel.MEDIUM),
            VisualElement(name="Building", importance=ImportanceLevel.MEDIUM),
            VisualElement(name="Tiny Sign", importance=ImportanceLevel.LOW),
        ],
        dominant_colors=ColorProfile(),
        composition=CompositionProfile(),
        visual_details=DetailProfile(),
    )


def test_perfect_score_invariants(sample_reference_profile):
    engine = WeightedScoringEngine()
    metrics = FeatureComparisonMetrics(
        semantic_similarity_raw=1.0,
        high_importance_found_ratio=1.0,
        medium_importance_found_ratio=1.0,
        low_importance_found_ratio=1.0,
        weighted_object_score_raw=1.0,
        focal_distance=0.0,
        quadrant_correlation=1.0,
        symmetry_match=1.0,
        composition_score_raw=1.0,
        color_palette_similarity=1.0,
        brightness_similarity=1.0,
        warmth_similarity=1.0,
        contrast_similarity=1.0,
        color_lighting_score_raw=1.0,
        edge_density_similarity=1.0,
        texture_similarity=1.0,
        fine_detail_score_raw=1.0,
    )

    scores = engine.calculate_scores(metrics, sample_reference_profile)

    assert scores.semantic_similarity == 30.0
    assert scores.object_accuracy == 25.0
    assert scores.composition_spatial == 20.0
    assert scores.color_lighting == 15.0
    assert scores.fine_details == 10.0
    assert scores.total_score == 100.0


def test_zero_score_invariants(sample_reference_profile):
    engine = WeightedScoringEngine()
    metrics = FeatureComparisonMetrics(
        semantic_similarity_raw=0.0,
        high_importance_found_ratio=0.0,
        medium_importance_found_ratio=0.0,
        low_importance_found_ratio=0.0,
        weighted_object_score_raw=0.0,
        focal_distance=1.0,
        quadrant_correlation=0.0,
        symmetry_match=0.0,
        composition_score_raw=0.0,
        color_palette_similarity=0.0,
        brightness_similarity=0.0,
        warmth_similarity=0.0,
        contrast_similarity=0.0,
        color_lighting_score_raw=0.0,
        edge_density_similarity=0.0,
        texture_similarity=0.0,
        fine_detail_score_raw=0.0,
    )

    scores = engine.calculate_scores(metrics, sample_reference_profile)

    assert scores.semantic_similarity == 0.0
    assert scores.object_accuracy == 0.0
    assert scores.composition_spatial == 0.0
    assert scores.color_lighting == 0.0
    assert scores.fine_details == 0.0
    assert scores.total_score == 0.0


def test_high_importance_penalty(sample_reference_profile):
    engine = WeightedScoringEngine()
    # Case A: 100% of high importance found
    metrics_a = FeatureComparisonMetrics(
        semantic_similarity_raw=0.8,
        high_importance_found_ratio=1.0,
        weighted_object_score_raw=0.8,
        composition_score_raw=0.8,
        color_lighting_score_raw=0.8,
        fine_detail_score_raw=0.8,
    )
    scores_a = engine.calculate_scores(metrics_a, sample_reference_profile)

    # Case B: Missed 50% of high importance items with same base raw score
    metrics_b = FeatureComparisonMetrics(
        semantic_similarity_raw=0.8,
        high_importance_found_ratio=0.5,
        weighted_object_score_raw=0.8,
        composition_score_raw=0.8,
        color_lighting_score_raw=0.8,
        fine_detail_score_raw=0.8,
    )
    scores_b = engine.calculate_scores(metrics_b, sample_reference_profile)

    assert scores_b.object_accuracy < scores_a.object_accuracy
    assert scores_b.total_score < scores_a.total_score
