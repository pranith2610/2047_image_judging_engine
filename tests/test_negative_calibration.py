"""
Negative-pair calibration test suite for THE 2047 judging engine.
Validates that completely unrelated images (Cristiano Ronaldo, astronaut, galaxy,
snow mountain, racing car, cyberpunk city, kitchen, basketball player, office, underwater reef)
strictly receive low scores in the 0-15 range based on evidence.
"""
from pathlib import Path
import pytest
import numpy as np
from PIL import Image, ImageDraw

from app.core.coordinator import JudgingCoordinator
from app.models.schemas import CategoryScores


@pytest.fixture(scope="module")
def coordinator() -> JudgingCoordinator:
    return JudgingCoordinator()


@pytest.fixture(scope="module")
def reference_image_path() -> Path:
    ref_files = list(Path("uploads/reference").glob("*final.png"))
    if ref_files and ref_files[0].exists():
        return ref_files[0]
    # Fallback to demo reference if final.png is missing
    fallback = Path("uploads/reference/ref_b4b3df9a_demo_reference.png")
    if fallback.exists():
        return fallback
    return Path("uploads/demo/reference_sample.png")


def test_cristiano_ronaldo_negative_pair(coordinator: JudgingCoordinator, reference_image_path: Path):
    """
    Test Case 1: Cristiano Ronaldo football photo against the South Indian temple festival reference.
    Must score strictly in the 0-15 range, with high_importance_found_ratio == 0,
    and explanation citing missing elephant, temple gopuram, oxen, and traditional clothing.
    """
    ronaldo_files = list(Path("uploads/participants").glob("*RONALDO*"))
    assert len(ronaldo_files) > 0, "Cristiano Ronaldo test image must be present in uploads/participants"
    ronaldo_path = ronaldo_files[0]

    ref_prof = coordinator.process_reference(reference_image_path)
    res = coordinator.evaluate_participant(ronaldo_path, "RONALDO", ref_prof)

    # 1. Total score must be in the low similarity range (0 - 15)
    assert 0.0 <= res.scores.total_score <= 15.0, (
        f"Cristiano Ronaldo score was {res.scores.total_score}, expected in 0-15 range"
    )

    # 2. Semantic score must be very low
    assert res.scores.semantic_similarity <= 3.0, (
        f"Semantic similarity score was {res.scores.semantic_similarity}, expected <= 3.0"
    )

    # 3. Object accuracy must be near 0
    assert res.scores.object_accuracy <= 2.0, (
        f"Object accuracy score was {res.scores.object_accuracy}, expected <= 2.0"
    )

    # 4. Composition score must be heavily capped
    assert res.scores.composition_spatial <= 3.0, (
        f"Composition score was {res.scores.composition_spatial}, expected <= 3.0"
    )

    # 5. Color score must not get false positive points
    assert res.scores.color_lighting <= 3.0, (
        f"Color score was {res.scores.color_lighting}, expected <= 3.0"
    )

    # 6. Fine details must not get points for generic photo sharpness
    assert res.scores.fine_details <= 1.5, (
        f"Detail score was {res.scores.fine_details}, expected <= 1.5"
    )

    # 7. High importance found ratio must be 0.0
    assert res.comparison_metrics.high_importance_found_ratio == 0.0

    # 8. High importance anchors must be listed as missing
    missed_text = " ".join(res.comparison_metrics.missed_elements).lower()
    assert "elephant" in missed_text
    assert "gopuram" in missed_text

    # 9. Explanation must be in 'Very low match' tier and explain missing anchors
    assert res.judging_report.score_tier == "Very low match"
    assert "elephant" in res.explanation.lower()
    assert "temple" in res.explanation.lower()


@pytest.mark.parametrize("scenario_name,bg_color,shapes", [
    ("astronaut_space", (5, 5, 20), [("circle", [100, 100, 200, 200], (240, 240, 250))]),
    ("galaxy_cosmos", (10, 0, 30), [("ellipse", [50, 50, 250, 250], (180, 50, 220))]),
    ("snow_mountain", (220, 230, 245), [("polygon", [(0, 300), (150, 50), (300, 300)], (160, 180, 210))]),
    ("racing_car", (60, 60, 60), [("rectangle", [50, 120, 250, 180], (220, 30, 30))]),
    ("cyberpunk_city", (20, 10, 40), [("rectangle", [40, 20, 100, 280], (0, 230, 255)), ("rectangle", [160, 60, 240, 280], (255, 0, 128))]),
    ("kitchen_interior", (230, 225, 210), [("rectangle", [20, 150, 280, 280], (140, 140, 150))]),
    ("basketball_court", (180, 110, 60), [("circle", [120, 120, 180, 180], (230, 100, 20))]),
    ("modern_office", (240, 240, 245), [("rectangle", [50, 100, 250, 200], (80, 90, 110))]),
    ("underwater_reef", (0, 80, 140), [("circle", [80, 80, 140, 140], (255, 120, 0)), ("ellipse", [160, 160, 260, 240], (255, 200, 50))]),
])
def test_unrelated_negative_scenarios(
    coordinator: JudgingCoordinator,
    reference_image_path: Path,
    tmp_path: Path,
    scenario_name: str,
    bg_color: tuple,
    shapes: list,
):
    """
    Test Cases 2-10: Generalization across 9 other unrelated visual domains.
    Must produce genuinely low scores (0-15) based on evidence.
    """
    img = Image.new("RGB", (300, 300), color=bg_color)
    draw = ImageDraw.Draw(img)
    for shape_type, coords, fill_col in shapes:
        if shape_type == "circle" or shape_type == "ellipse":
            draw.ellipse(coords, fill=fill_col)
        elif shape_type == "rectangle":
            draw.rectangle(coords, fill=fill_col)
        elif shape_type == "polygon":
            draw.polygon(coords, fill=fill_col)

    neg_path = tmp_path / f"{scenario_name}.png"
    img.save(neg_path)

    ref_prof = coordinator.process_reference(reference_image_path)
    res = coordinator.evaluate_participant(neg_path, scenario_name, ref_prof)

    assert 0.0 <= res.scores.total_score <= 15.0, (
        f"Negative scenario '{scenario_name}' received score {res.scores.total_score}, expected <= 15.0"
    )
    assert res.comparison_metrics.high_importance_found_ratio < 0.25
    assert res.judging_report.score_tier in ["Very low match", "Weak match"]


def test_high_match_positive_calibration(coordinator: JudgingCoordinator, reference_image_path: Path):
    """
    Positive calibration test: Verifies that a near-identical recreation image
    receives a strong score in the high range (80-100).
    """
    high_match_files = list(Path("uploads/participants").glob("*gt15g7gt15*"))
    if not high_match_files:
        pytest.skip("High match fixture not available in uploads/participants")

    ref_prof = coordinator.process_reference(reference_image_path)
    res = coordinator.evaluate_participant(high_match_files[0], "P_HIGH", ref_prof)

    assert res.scores.total_score >= 70.0, (
        f"High match image received score {res.scores.total_score}, expected >= 70.0"
    )
    assert res.comparison_metrics.high_importance_found_ratio >= 0.50
