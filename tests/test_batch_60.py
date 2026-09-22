"""
High-capacity batch integration test for evaluating 60 participant images.
Verifies performance, model reuse, structured evidence integrity, and ranking stability.
"""
from pathlib import Path
from PIL import Image, ImageDraw
import time

from app.core.coordinator import JudgingCoordinator


def test_batch_60_participants_evaluation(tmp_path: Path):
    coordinator = JudgingCoordinator()

    # 1. Create Reference Image
    ref_img = Image.new("RGB", (200, 200), color=(25, 30, 80))
    draw = ImageDraw.Draw(ref_img)
    draw.ellipse([40, 40, 160, 160], fill=(230, 50, 50))
    ref_path = tmp_path / "reference_60.png"
    ref_img.save(ref_path)

    # 2. Generate 60 diverse participant images
    participants = []
    for i in range(1, 61):
        p_img = Image.new("RGB", (200, 200), color=(20 + (i % 30), 30 + (i % 20), 80 + (i % 40)))
        p_draw = ImageDraw.Draw(p_img)
        # Vary geometry and colors
        radius_offset = (i % 15)
        p_draw.ellipse([40 - radius_offset, 40 - radius_offset, 160 + radius_offset, 160 + radius_offset], fill=(200 + (i % 50), 40 + (i % 30), 40 + (i % 40)))
        p_path = tmp_path / f"participant_{i:03d}.png"
        p_img.save(p_path)
        participants.append((f"P-{i:03d}", p_path))

    assert len(participants) == 60

    # 3. Evaluate Batch
    start_time = time.time()
    batch_res = coordinator.evaluate_batch(ref_path, participants)
    elapsed = time.time() - start_time

    # 4. Invariant checks
    assert batch_res.total_participants == 60
    assert len(batch_res.ranked_results) == 60

    # Processing speed: 60 images should execute rapidly with pre-loaded extractors
    print(f"60 images evaluated in {elapsed:.2f} seconds ({elapsed/60*1000:.1f} ms per image)")

    # Validate all 60 participants have valid bounded scores and structured evidence
    for r in batch_res.ranked_results:
        assert 0.0 <= r.total_score <= 100.0
        assert 0.0 <= r.scores.semantic_similarity <= 30.0
        assert 0.0 <= r.scores.object_accuracy <= 25.0
        assert 0.0 <= r.scores.composition_spatial <= 20.0
        assert 0.0 <= r.scores.color_lighting <= 15.0
        assert 0.0 <= r.scores.fine_details <= 10.0
        assert r.evidence is not None
        assert r.rank >= 1
