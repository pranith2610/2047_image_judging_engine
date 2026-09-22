"""
Comprehensive integration tests for the 6 official comparison scenarios in THE 2047:
1. Participant image nearly identical to reference.
2. Participant image with same objects but different arrangement.
3. Participant image with similar concept but different objects.
4. Participant image with correct objects but different colors.
5. Participant image with major objects missing.
6. Completely unrelated image.
"""
from pathlib import Path
from PIL import Image, ImageDraw
import pytest

from app.core.coordinator import JudgingCoordinator


def create_reference_scene(path: Path) -> Path:
    """
    Creates reference image:
    - Dark Navy background (20, 20, 70)
    - Major central red circle (240, 40, 40) at [50, 50, 250, 250]
    - Secondary yellow rectangle (240, 200, 30) in top-right [220, 20, 280, 80]
    - Peripheral green bar (30, 200, 50) in bottom-left [20, 220, 80, 280]
    """
    img = Image.new("RGB", (300, 300), color=(20, 20, 70))
    draw = ImageDraw.Draw(img)
    draw.ellipse([50, 50, 250, 250], fill=(240, 40, 40))
    draw.rectangle([220, 20, 280, 80], fill=(240, 200, 30))
    draw.rectangle([20, 220, 80, 280], fill=(30, 200, 50))
    img.save(path)
    return path


def test_scenario_1_nearly_identical(tmp_path: Path):
    """Scenario 1: Participant image nearly identical to reference."""
    ref_path = create_reference_scene(tmp_path / "ref.png")
    
    # Near perfect recreation with tiny minor pixel variance
    p1_path = tmp_path / "p1_identical.png"
    img = Image.new("RGB", (300, 300), color=(22, 22, 72))
    draw = ImageDraw.Draw(img)
    draw.ellipse([52, 52, 248, 248], fill=(238, 42, 42))
    draw.rectangle([218, 22, 278, 82], fill=(238, 198, 32))
    draw.rectangle([22, 218, 82, 278], fill=(32, 198, 52))
    img.save(p1_path)

    coordinator = JudgingCoordinator()
    ref_profile = coordinator.process_reference(ref_path)
    res = coordinator.evaluate_participant(p1_path, "P-IDENTICAL", ref_profile)

    # Score should be very high (~90-100)
    assert res.scores.total_score >= 88.0
    assert res.scores.semantic_similarity >= 25.0
    assert res.scores.object_accuracy >= 20.0
    assert res.scores.composition_spatial >= 16.0
    assert res.scores.color_lighting >= 12.0
    assert res.scores.fine_details >= 7.0
    assert res.evidence is not None
    assert len(res.evidence.matched_elements) > 0


def test_scenario_2_same_objects_different_arrangement(tmp_path: Path):
    """Scenario 2: Same objects but spatial arrangement inverted / shifted."""
    ref_path = create_reference_scene(tmp_path / "ref.png")
    
    p2_path = tmp_path / "p2_different_composition.png"
    # Same Navy background and same objects, but red circle moved to left, yellow moved to bottom
    img = Image.new("RGB", (300, 300), color=(20, 20, 70))
    draw = ImageDraw.Draw(img)
    # Moved main circle to upper-left corner
    draw.ellipse([10, 10, 150, 150], fill=(240, 40, 40))
    # Moved yellow rectangle to bottom-center
    draw.rectangle([120, 220, 180, 280], fill=(240, 200, 30))
    # Moved green bar to top-right
    draw.rectangle([220, 20, 280, 80], fill=(30, 200, 50))
    img.save(p2_path)

    coordinator = JudgingCoordinator()
    ref_profile = coordinator.process_reference(ref_path)

    # Also evaluate identical for direct comparison
    p1_path = tmp_path / "p1_identical.png"
    if not p1_path.exists():
        test_scenario_1_nearly_identical(tmp_path)
    res_p1 = coordinator.evaluate_participant(p1_path, "P1", ref_profile)
    res_p2 = coordinator.evaluate_participant(p2_path, "P2_SHIFTED", ref_profile)

    # Composition score for shifted arrangement must be lower than identical
    assert res_p2.scores.composition_spatial < res_p1.scores.composition_spatial
    assert len(res_p2.evidence.composition_differences) > 0


def test_scenario_3_similar_concept_different_objects(tmp_path: Path):
    """Scenario 3: Similar concept (dark scene + warm tones) but different objects (stars, diagonals)."""
    ref_path = create_reference_scene(tmp_path / "ref.png")
    
    p3_path = tmp_path / "p3_different_objects.png"
    img = Image.new("RGB", (300, 300), color=(30, 25, 65))
    draw = ImageDraw.Draw(img)
    # Starburst / diagonal lines instead of circles/rectangles
    draw.line([30, 30, 270, 270], fill=(240, 100, 30), width=15)
    draw.line([270, 30, 30, 270], fill=(240, 100, 30), width=15)
    draw.polygon([(150, 20), (180, 100), (260, 100), (200, 150), (220, 230), (150, 180)], fill=(200, 180, 40))
    img.save(p3_path)

    coordinator = JudgingCoordinator()
    ref_profile = coordinator.process_reference(ref_path)
    res = coordinator.evaluate_participant(p3_path, "P3_CONCEPT", ref_profile)

    # Moderate semantic score, but lower object accuracy score
    assert 0.0 <= res.scores.total_score <= 100.0
    assert res.scores.semantic_similarity > 10.0
    assert len(res.evidence.missing_elements) > 0 or len(res.evidence.unexpected_elements) > 0


def test_scenario_4_correct_objects_different_colors(tmp_path: Path):
    """Scenario 4: Correct objects and layout, but completely different color palette (cool icy cyan/white vs warm red/navy)."""
    ref_path = create_reference_scene(tmp_path / "ref.png")
    
    p4_path = tmp_path / "p4_different_colors.png"
    # Bright white/ice background with cyan/purple shapes
    img = Image.new("RGB", (300, 300), color=(240, 248, 255))
    draw = ImageDraw.Draw(img)
    draw.ellipse([50, 50, 250, 250], fill=(0, 200, 240))      # Cyan circle instead of red
    draw.rectangle([220, 20, 280, 80], fill=(180, 40, 240))   # Purple instead of yellow
    draw.rectangle([20, 220, 80, 280], fill=(40, 40, 120))    # Dark blue instead of green
    img.save(p4_path)

    coordinator = JudgingCoordinator()
    ref_profile = coordinator.process_reference(ref_path)
    res = coordinator.evaluate_participant(p4_path, "P4_COLORS", ref_profile)

    # Color/lighting score should be noticeably lower
    assert res.scores.color_lighting < 11.0
    assert len(res.evidence.color_differences) > 0


def test_scenario_5_major_objects_missing(tmp_path: Path):
    """Scenario 5: Major HIGH-importance object (central red circle) missing completely."""
    ref_path = create_reference_scene(tmp_path / "ref.png")
    
    p5_path = tmp_path / "p5_missing_main.png"
    # Navy background, only minor peripheral elements drawn
    img = Image.new("RGB", (300, 300), color=(20, 20, 70))
    draw = ImageDraw.Draw(img)
    # NO central circle!
    draw.rectangle([220, 20, 280, 80], fill=(240, 200, 30))
    draw.rectangle([20, 220, 80, 280], fill=(30, 200, 50))
    img.save(p5_path)

    coordinator = JudgingCoordinator()
    ref_profile = coordinator.process_reference(ref_path)
    res = coordinator.evaluate_participant(p5_path, "P5_MISSING", ref_profile)

    # Object score should suffer significant deduction
    assert res.scores.object_accuracy < 18.0
    assert any("HIGH" in m for m in res.evidence.missing_elements)


def test_scenario_6_completely_unrelated(tmp_path: Path):
    """Scenario 6: Completely unrelated image (Solid white with single grey line)."""
    ref_path = create_reference_scene(tmp_path / "ref.png")
    
    p6_path = tmp_path / "p6_unrelated.png"
    img = Image.new("RGB", (300, 300), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.line([0, 0, 10, 10], fill=(128, 128, 128), width=1)
    img.save(p6_path)

    coordinator = JudgingCoordinator()
    ref_profile = coordinator.process_reference(ref_path)
    res = coordinator.evaluate_participant(p6_path, "P6_UNRELATED", ref_profile)

    # Total score should be very low
    assert res.scores.total_score < 45.0
    assert res.scores.object_accuracy < 10.0
    assert res.scores.semantic_similarity < 18.0
