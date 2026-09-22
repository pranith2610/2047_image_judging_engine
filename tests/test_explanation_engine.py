"""
Tests for the Explanation Engine and judging report generation.
Verifies score consistency, anti-hallucination, score-tier interpretation,
strengths/differences synthesis, key elements analysis, and CSV/JSON exports.
"""
from pathlib import Path
import pytest
from PIL import Image, ImageDraw

from app.models.schemas import (
    ReferenceProfile,
    ParticipantProfile,
    CategoryScores,
    FeatureComparisonMetrics,
    JudgingReport,
    DominantColor,
    ColorFeature,
    ObjectFeature,
    CompositionFeature,
    LightingFeature,
    DetailFeature,
)
from app.models.enums import ImportanceLevel, ObjectMatchStatus
from app.core.explanations.engine import RuleBasedExplanationEngine
from app.core.coordinator import JudgingCoordinator
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def explanation_engine():
    return RuleBasedExplanationEngine()


@pytest.fixture
def mock_reference_profile():
    return ReferenceProfile(
        image_id="REF-TEST",
        filename="ref.png",
        dimensions=(400, 400),
        colors=ColorFeature(
            dominant_colors=[DominantColor(hex_code="#C82828", rgb=(200, 40, 40), percentage=40.0)],
            warmth=0.6,
            average_brightness=0.5,
        ),
        objects=[
            ObjectFeature(name="Crimson Central Subject", importance=ImportanceLevel.HIGH, location="center"),
            ObjectFeature(name="Amber Secondary Box", importance=ImportanceLevel.MEDIUM, location="upper-right"),
            ObjectFeature(name="Distant Lamp Post", importance=ImportanceLevel.LOW, location="lower-left"),
        ],
        composition=CompositionFeature(
            focal_center=(0.5, 0.5),
            layout_description="Centered anchor composition",
        ),
        lighting=LightingFeature(lighting_description="Warm sunset glow"),
        details=DetailFeature(),
    )


def test_score_tier_mapping(explanation_engine):
    """Verify all 7 score tiers match the exact requirements."""
    assert explanation_engine.get_score_tier(95.0) == "Exceptional match"
    assert explanation_engine.get_score_tier(90.0) == "Exceptional match"
    assert explanation_engine.get_score_tier(85.0) == "Very strong match"
    assert explanation_engine.get_score_tier(80.0) == "Very strong match"
    assert explanation_engine.get_score_tier(75.0) == "Strong match"
    assert explanation_engine.get_score_tier(70.0) == "Strong match"
    assert explanation_engine.get_score_tier(65.0) == "Moderate match"
    assert explanation_engine.get_score_tier(60.0) == "Moderate match"
    assert explanation_engine.get_score_tier(55.0) == "Partial match"
    assert explanation_engine.get_score_tier(50.0) == "Partial match"
    assert explanation_engine.get_score_tier(45.0) == "Weak match"
    assert explanation_engine.get_score_tier(40.0) == "Weak match"
    assert explanation_engine.get_score_tier(35.0) == "Very low match"
    assert explanation_engine.get_score_tier(0.0) == "Very low match"


def test_high_score_explanation_consistency(explanation_engine, mock_reference_profile):
    """Near-perfect score (94/100) must produce positive, consistent explanations with no false missing elements."""
    scores = CategoryScores(
        semantic_similarity=28.5,
        object_accuracy=24.0,
        composition_spatial=19.0,
        color_lighting=14.0,
        fine_details=8.5,
        total_score=94.0,
    )
    metrics = FeatureComparisonMetrics(
        semantic_similarity_raw=0.95,
        high_importance_found_ratio=1.0,
        medium_importance_found_ratio=1.0,
        low_importance_found_ratio=1.0,
        weighted_object_score_raw=0.96,
        matched_elements=["Crimson Central Subject (PRESENT)", "Amber Secondary Box (PRESENT)", "Distant Lamp Post (PRESENT)"],
        focal_distance=0.04,
        color_palette_similarity=0.93,
        brightness_similarity=0.95,
        edge_density_similarity=0.88,
    )
    part_profile = ParticipantProfile(
        participant_id="P_EXCEPTIONAL",
        filename="p_high.png",
        dimensions=(400, 400),
    )

    report = explanation_engine.generate_judging_report(
        mock_reference_profile, part_profile, metrics, scores
    )

    assert report.score_tier == "Exceptional match"
    assert report.final_score == 94.0
    assert "exceptionally well" in report.overall_explanation.lower() or "very well" in report.overall_explanation.lower()
    
    # Assert all 5 categories have non-empty explanations
    assert len(report.category_explanations) == 5
    assert "28.5/30" in report.category_explanations["semantic_similarity"]
    assert "24.0/25" in report.category_explanations["object_accuracy"]

    # Check strengths
    assert len(report.strengths) > 0
    assert all(s.startswith("✓") for s in report.strengths)

    # Check key elements status
    assert len(report.key_elements) == 3
    for ke in report.key_elements:
        assert ke.status == ObjectMatchStatus.PRESENT


def test_low_score_anti_contradiction(explanation_engine, mock_reference_profile):
    """A low score (32/100) must NEVER claim reproduction was successful or exceptional."""
    scores = CategoryScores(
        semantic_similarity=5.0,
        object_accuracy=6.0,
        composition_spatial=8.0,
        color_lighting=7.0,
        fine_details=6.0,
        total_score=32.0,
    )
    metrics = FeatureComparisonMetrics(
        semantic_similarity_raw=0.15,
        high_importance_found_ratio=0.0,
        medium_importance_found_ratio=0.0,
        low_importance_found_ratio=0.0,
        weighted_object_score_raw=0.24,
        missed_elements=["Crimson Central Subject (MISSING)", "Amber Secondary Box (MISSING)"],
        focal_distance=0.55,
        color_palette_similarity=0.30,
    )
    part_profile = ParticipantProfile(
        participant_id="P_LOW",
        filename="p_low.png",
        dimensions=(400, 400),
    )

    report = explanation_engine.generate_judging_report(
        mock_reference_profile, part_profile, metrics, scores
    )

    assert report.score_tier == "Very low match"
    assert report.final_score == 32.0

    # NO CONTRADICTION RULE: Must not contain positive praise
    overall_lower = report.overall_explanation.lower()
    assert "exceptional" not in overall_lower
    assert "accurately reproduced almost everything" not in overall_lower
    assert "diverges significantly" in overall_lower

    # Missing HIGH-importance object must be flagged in differences
    diffs_joined = " ".join(report.differences)
    assert "Crimson Central Subject" in diffs_joined or "anchor missing" in diffs_joined


def test_category_strengths_and_weaknesses_ranking(explanation_engine):
    """Test that category rankings are sorted by % of max points earned."""
    # Semantic: 27/30 (90%), Object: 15/25 (60%), Comp: 18/20 (90%), Color: 6/15 (40%), Detail: 9/10 (90%)
    scores = CategoryScores(
        semantic_similarity=27.0,
        object_accuracy=15.0,
        composition_spatial=18.0,
        color_lighting=6.0,
        fine_details=9.0,
        total_score=75.0,
    )
    strongest, to_improve = explanation_engine._rank_category_performance(scores)

    # Color (40%) and Object (60%) must be in areas to improve
    assert "Color & Lighting" in to_improve
    assert "Object Accuracy" in to_improve

    # Top categories should be in strongest
    assert "Semantic Similarity" in strongest or "Fine Details" in strongest or "Composition" in strongest


def test_end_to_end_scenarios_explanation(tmp_path: Path):
    """Verify that evaluations through JudgingCoordinator generate complete, consistent judging reports."""
    # Create reference
    ref_path = tmp_path / "ref.png"
    img = Image.new("RGB", (300, 300), color=(15, 20, 50))
    draw = ImageDraw.Draw(img)
    draw.ellipse([80, 80, 220, 220], fill=(220, 30, 30))
    img.save(ref_path)

    # Participant 1: Identical
    p1_path = tmp_path / "p1.png"
    img.save(p1_path)

    # Participant 2: Completely unrelated
    p2_path = tmp_path / "p2.png"
    img_unrel = Image.new("RGB", (300, 300), color=(240, 240, 240))
    img_unrel.save(p2_path)

    coordinator = JudgingCoordinator()
    batch_res = coordinator.evaluate_batch(ref_path, [("P1", p1_path), ("P2", p2_path)])

    assert len(batch_res.ranked_results) == 2
    r1 = batch_res.ranked_results[0]
    r2 = batch_res.ranked_results[1]

    assert r1.participant_id == "P1"
    assert r1.total_score >= 90.0
    assert r1.judging_report is not None
    assert r1.judging_report.score_tier == "Exceptional match"
    assert len(r1.judging_report.strengths) > 0

    assert r2.participant_id == "P2"
    assert r2.total_score < 45.0
    assert r2.judging_report is not None
    assert r2.judging_report.score_tier in ("Very low match", "Weak match")
    assert "diverges significantly" in r2.judging_report.overall_explanation.lower() or "partial" in r2.judging_report.overall_explanation.lower()


def test_csv_and_json_export_routes(tmp_path: Path):
    """Verify CSV and JSON export API endpoints."""
    ref_path = tmp_path / "ref.png"
    img = Image.new("RGB", (200, 200), color=(20, 30, 40))
    img.save(ref_path)

    p_path = tmp_path / "p.png"
    img.save(p_path)

    coordinator = JudgingCoordinator()
    batch_res = coordinator.evaluate_batch(ref_path, [("P01", p_path)])

    client = TestClient(app)

    # Test CSV Export
    csv_resp = client.post("/api/export/csv", json=batch_res.model_dump())
    assert csv_resp.status_code == 200
    assert "text/csv" in csv_resp.headers["content-type"]
    assert "P01" in csv_resp.text
    assert "Final Score" in csv_resp.text

    # Test JSON Export
    json_resp = client.post("/api/export/json", json=batch_res.model_dump())
    assert json_resp.status_code == 200
    assert "application/json" in json_resp.headers["content-type"]
    data = json_resp.json()
    assert data["total_participants"] == 1
    assert data["ranked_results"][0]["participant_id"] == "P01"
    assert "judging_report" in data["ranked_results"][0]
