"""
End-to-end integration test for the THE 2047 judging pipeline.
"""
import io
from pathlib import Path
from PIL import Image, ImageDraw
from app.core.coordinator import JudgingCoordinator


def generate_synthetic_image(bg_color=(30, 30, 80), shape_color=(240, 50, 50), shape="circle") -> Image.Image:
    img = Image.new("RGB", (300, 300), color=bg_color)
    draw = ImageDraw.Draw(img)
    if shape == "circle":
        draw.ellipse([80, 80, 220, 220], fill=shape_color)
    elif shape == "rect":
        draw.rectangle([80, 80, 220, 220], fill=shape_color)
    elif shape == "line":
        draw.line([30, 30, 270, 270], fill=shape_color, width=10)
    return img


def test_end_to_end_judging_batch(tmp_path: Path):
    coordinator = JudgingCoordinator()

    # 1. Create Reference Image (Dark blue background + Red central circle)
    ref_img = generate_synthetic_image(bg_color=(20, 20, 70), shape_color=(240, 40, 40), shape="circle")
    ref_path = tmp_path / "reference_test.png"
    ref_img.save(ref_path)

    # 2. Create 3 Participant Recreations
    # P1: Exact / highest fidelity recreation (same colors, same circle geometry)
    p1_img = generate_synthetic_image(bg_color=(20, 20, 70), shape_color=(240, 40, 40), shape="circle")
    p1_path = tmp_path / "p1_test.png"
    p1_img.save(p1_path)

    # P2: Moderate recreation (Different shape - rectangle, shifted color)
    p2_img = generate_synthetic_image(bg_color=(50, 50, 100), shape_color=(200, 80, 80), shape="rect")
    p2_path = tmp_path / "p2_test.png"
    p2_img.save(p2_path)

    # P3: Poor recreation (Inverted colors, thin line)
    p3_img = generate_synthetic_image(bg_color=(240, 240, 240), shape_color=(10, 10, 10), shape="line")
    p3_path = tmp_path / "p3_test.png"
    p3_img.save(p3_path)

    participants = [
        ("PARTICIPANT-01", p1_path),
        ("PARTICIPANT-02", p2_path),
        ("PARTICIPANT-03", p3_path),
    ]

    # 3. Execute Batch
    batch_res = coordinator.evaluate_batch(ref_path, participants)

    # 4. Invariant Assertions
    assert batch_res.total_participants == 3
    assert len(batch_res.ranked_results) == 3

    # Reference profile checks
    assert batch_res.reference_profile.dimensions == (300, 300)
    assert len(batch_res.reference_profile.dominant_colors.dominant_colors) > 0

    # Ranking checks
    ranks = [r.rank for r in batch_res.ranked_results]
    assert ranks == [1, 2, 3]

    # Scores must be descending
    scores = [r.total_score for r in batch_res.ranked_results]
    assert scores == sorted(scores, reverse=True)

    # P1 (most faithful recreation) should be ranked #1
    top_participant = batch_res.ranked_results[0]
    assert top_participant.participant_id == "PARTICIPANT-01"

    # Verify score bounds for all participants
    for r in batch_res.ranked_results:
        assert 0.0 <= r.total_score <= 100.0
        assert 0.0 <= r.scores.semantic_similarity <= 30.0
        assert 0.0 <= r.scores.object_accuracy <= 25.0
        assert 0.0 <= r.scores.composition_spatial <= 20.0
        assert 0.0 <= r.scores.color_lighting <= 15.0
        assert 0.0 <= r.scores.fine_details <= 10.0
        assert len(r.explanation) > 10
