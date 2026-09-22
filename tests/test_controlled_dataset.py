"""
Controlled Test Dataset and Comprehensive QA Suite for THE 2047 Engine.
Tests all canonical evaluation scenarios (Tests A through J), sanity ordering,
score component boundaries, contradiction detection, spatial understanding,
duplicate detection, failure recovery, session resume, and 52/60 participant dry runs.
"""
import pytest
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from pathlib import Path
import tempfile
import time
import tracemalloc

from app.models.schemas import (
    ReferenceProfile,
    ParticipantProfile,
    CategoryScores,
    JudgingReport,
    ImportanceLevel,
    ObjectMatchStatus,
)
from app.core.analyzers.reference import BaselineReferenceAnalyzer
from app.core.analyzers.participant import BaselineParticipantAnalyzer
from app.core.comparators.comparator import BaselineFeatureComparator
from app.core.scoring.engine import WeightedScoringEngine
from app.core.explanations.engine import RuleBasedExplanationEngine, validate_explanation_consistency
from app.core.ranking.engine import StandardRankingEngine
from app.core.coordinator import JudgingCoordinator
from app.config import CATEGORY_WEIGHTS, TOTAL_POINTS


@pytest.fixture(scope="module")
def coordinator():
    return JudgingCoordinator()


@pytest.fixture(scope="module")
def test_dataset(tmp_path_factory):
    """
    Generate synthetic test dataset covering Tests A through J with controlled visual properties.
    """
    tmp_dir = tmp_path_factory.mktemp("controlled_data")

    # 1. BASELINE REFERENCE IMAGE:
    # Blue sky background, green grass bottom, red car (center), yellow sun (upper right), person (left)
    ref_img = Image.new("RGB", (512, 512), (135, 206, 235))  # Sky blue
    draw = ImageDraw.Draw(ref_img)
    draw.rectangle([0, 350, 512, 512], fill=(34, 139, 34))   # Grass green
    draw.ellipse([380, 40, 460, 120], fill=(255, 220, 0))    # Sun (top right)
    draw.rectangle([180, 260, 340, 360], fill=(220, 20, 60)) # Red car (center) - HIGH importance
    draw.rectangle([60, 280, 110, 380], fill=(0, 0, 128))    # Person (left) - HIGH importance
    draw.rectangle([420, 250, 470, 350], fill=(139, 69, 19)) # Tree trunk (right) - LOW importance
    draw.ellipse([400, 180, 490, 270], fill=(0, 100, 0))     # Tree foliage (right) - LOW importance
    ref_path = tmp_dir / "reference.png"
    ref_img.save(ref_path)

    # TEST A: Nearly identical image (slight pixel noise)
    test_a = ref_img.copy()
    test_a_path = tmp_dir / "test_a_identical.png"
    test_a.save(test_a_path)

    # TEST B: Same scene with minor changes (sun slightly shifted, tree slightly wider)
    test_b = Image.new("RGB", (512, 512), (130, 200, 230))
    draw_b = ImageDraw.Draw(test_b)
    draw_b.rectangle([0, 350, 512, 512], fill=(35, 140, 35))
    draw_b.ellipse([360, 50, 440, 130], fill=(250, 215, 0))   # Sun slightly shifted
    draw_b.rectangle([185, 265, 345, 365], fill=(215, 25, 55)) # Red car
    draw_b.rectangle([65, 285, 115, 385], fill=(10, 10, 135))  # Person
    draw_b.rectangle([415, 250, 475, 350], fill=(139, 69, 19)) # Tree
    draw_b.ellipse([395, 180, 495, 270], fill=(0, 100, 0))
    test_b_path = tmp_dir / "test_b_minor_changes.png"
    test_b.save(test_b_path)

    # TEST C: Same major objects but different composition (Car left, Person right, Sun lower-left)
    test_c = Image.new("RGB", (512, 512), (135, 206, 235))
    draw_c = ImageDraw.Draw(test_c)
    draw_c.rectangle([0, 350, 512, 512], fill=(34, 139, 34))
    draw_c.ellipse([50, 360, 130, 440], fill=(255, 220, 0))    # Sun in lower left (inverted)
    draw_c.rectangle([60, 260, 220, 360], fill=(220, 20, 60))  # Red car on LEFT
    draw_c.rectangle([380, 280, 430, 380], fill=(0, 0, 128))   # Person on RIGHT
    draw_c.rectangle([240, 250, 290, 350], fill=(139, 69, 19)) # Tree in CENTER
    draw_c.ellipse([220, 180, 310, 270], fill=(0, 100, 0))
    test_c_path = tmp_dir / "test_c_diff_composition.png"
    test_c.save(test_c_path)

    # TEST D: Same composition but major object missing (NO CAR, only person and tree)
    test_d = Image.new("RGB", (512, 512), (135, 206, 235))
    draw_d = ImageDraw.Draw(test_d)
    draw_d.rectangle([0, 350, 512, 512], fill=(34, 139, 34))
    draw_d.ellipse([380, 40, 460, 120], fill=(255, 220, 0))
    # Car omitted!
    draw_d.rectangle([60, 280, 110, 380], fill=(0, 0, 128))    # Person
    draw_d.rectangle([420, 250, 470, 350], fill=(139, 69, 19)) # Tree
    draw_d.ellipse([400, 180, 490, 270], fill=(0, 100, 0))
    test_d_path = tmp_dir / "test_d_missing_car.png"
    test_d.save(test_d_path)

    # TEST E: Same objects and composition but substantially different colors (Dark night theme, purple grass)
    test_e = Image.new("RGB", (512, 512), (15, 15, 40))        # Dark navy sky
    draw_e = ImageDraw.Draw(test_e)
    draw_e.rectangle([0, 350, 512, 512], fill=(80, 20, 90))    # Dark purple ground
    draw_e.ellipse([380, 40, 460, 120], fill=(200, 200, 255))  # White moon
    draw_e.rectangle([180, 260, 340, 360], fill=(0, 200, 200)) # Cyan car instead of red
    draw_e.rectangle([60, 280, 110, 380], fill=(255, 140, 0))  # Orange person
    draw_e.rectangle([420, 250, 470, 350], fill=(60, 60, 60))
    draw_e.ellipse([400, 180, 490, 270], fill=(120, 50, 150))
    test_e_path = tmp_dir / "test_e_diff_colors.png"
    test_e.save(test_e_path)

    # TEST F: Similar concept but different scene (Desert landscape with yellow sand and orange sky)
    test_f = Image.new("RGB", (512, 512), (255, 140, 50))      # Sunset orange sky
    draw_f = ImageDraw.Draw(test_f)
    draw_f.rectangle([0, 320, 512, 512], fill=(237, 201, 175)) # Sand dunes
    draw_f.ellipse([200, 80, 280, 160], fill=(255, 240, 150))  # Sun center
    draw_f.rectangle([220, 340, 300, 400], fill=(139, 69, 19)) # Camel/box
    test_f_path = tmp_dir / "test_f_similar_concept.png"
    test_f.save(test_f_path)

    # TEST G: Completely unrelated image (High-tech purple circuit board grid)
    test_g = Image.new("RGB", (512, 512), (50, 0, 80))
    draw_g = ImageDraw.Draw(test_g)
    for x in range(0, 512, 32):
        draw_g.line([(x, 0), (x, 512)], fill=(180, 0, 255), width=2)
    for y in range(0, 512, 32):
        draw_g.line([(0, y), (512, y)], fill=(0, 255, 200), width=2)
    test_g_path = tmp_dir / "test_g_unrelated.png"
    test_g.save(test_g_path)

    # TEST H: Same image resized and compressed (256x256 JPEG at 60% quality)
    test_h = ref_img.resize((256, 256), Image.Resampling.LANCZOS)
    test_h_path = tmp_dir / "test_h_compressed.jpg"
    test_h.save(test_h_path, "JPEG", quality=60)

    # TEST I: Same scene with artistic filtering (Edge enhancement / sketch-like)
    test_i = ref_img.filter(ImageFilter.EDGE_ENHANCE_MORE)
    test_i_path = tmp_dir / "test_i_artistic.png"
    test_i.save(test_i_path)

    # TEST J-1 (Participant A: Keeps Car + Person, misses tree)
    test_j1 = Image.new("RGB", (512, 512), (135, 206, 235))
    draw_j1 = ImageDraw.Draw(test_j1)
    draw_j1.rectangle([0, 350, 512, 512], fill=(34, 139, 34))
    draw_j1.ellipse([380, 40, 460, 120], fill=(255, 220, 0))
    draw_j1.rectangle([180, 260, 340, 360], fill=(220, 20, 60)) # Car (HIGH)
    draw_j1.rectangle([60, 280, 110, 380], fill=(0, 0, 128))    # Person (HIGH)
    # Tree omitted
    test_j1_path = tmp_dir / "test_j1_keeps_high.png"
    test_j1.save(test_j1_path)

    # TEST J-2 (Participant B: Keeps Tree, misses Car and Person)
    test_j2 = Image.new("RGB", (512, 512), (135, 206, 235))
    draw_j2 = ImageDraw.Draw(test_j2)
    draw_j2.rectangle([0, 350, 512, 512], fill=(34, 139, 34))
    draw_j2.ellipse([380, 40, 460, 120], fill=(255, 220, 0))
    # Car & Person omitted
    draw_j2.rectangle([420, 250, 470, 350], fill=(139, 69, 19)) # Tree (LOW)
    draw_j2.ellipse([400, 180, 490, 270], fill=(0, 100, 0))
    test_j2_path = tmp_dir / "test_j2_keeps_low.png"
    test_j2.save(test_j2_path)

    return {
        "dir": tmp_dir,
        "ref": ref_path,
        "test_a": test_a_path,
        "test_b": test_b_path,
        "test_c": test_c_path,
        "test_d": test_d_path,
        "test_e": test_e_path,
        "test_f": test_f_path,
        "test_g": test_g_path,
        "test_h": test_h_path,
        "test_i": test_i_path,
        "test_j1": test_j1_path,
        "test_j2": test_j2_path,
    }


# ==========================================
# 1. CONTROLLED SCENARIO TESTS (TESTS A - J)
# ==========================================

def test_scenario_a_nearly_identical(coordinator, test_dataset):
    """TEST A: Nearly identical image -> Very high score (>= 88.0)."""
    ref_prof = coordinator.process_reference(test_dataset["ref"])
    res = coordinator.evaluate_participant(test_dataset["test_a"], "P_A", ref_prof)

    assert res.scores.total_score >= 88.0, f"Expected >= 88.0, got {res.scores.total_score}"
    assert res.scores.semantic_similarity >= 27.0
    assert res.scores.object_accuracy >= 22.0
    assert res.scores.composition_spatial >= 18.0
    assert res.scores.color_lighting >= 13.0


def test_scenario_b_minor_changes(coordinator, test_dataset):
    """TEST B: Same scene with minor changes -> High score (>= 75.0)."""
    ref_prof = coordinator.process_reference(test_dataset["ref"])
    res = coordinator.evaluate_participant(test_dataset["test_b"], "P_B", ref_prof)

    assert res.scores.total_score >= 75.0, f"Expected >= 75.0, got {res.scores.total_score}"
    assert res.scores.semantic_similarity >= 24.0


def test_scenario_c_composition_shift(coordinator, test_dataset):
    """
    TEST C: Same major objects but inverted/different composition.
    Expected: Lower composition score than Test A/B, but high semantic and object scores.
    """
    ref_prof = coordinator.process_reference(test_dataset["ref"])
    res_a = coordinator.evaluate_participant(test_dataset["test_a"], "P_A", ref_prof)
    res_c = coordinator.evaluate_participant(test_dataset["test_c"], "P_C", ref_prof)

    # Composition score in C should drop noticeably compared to A
    assert res_c.scores.composition_spatial < res_a.scores.composition_spatial
    assert res_c.scores.composition_spatial <= 16.5
    # Object score should remain respectable since major objects are present
    assert res_c.scores.object_accuracy >= 14.0


def test_scenario_d_missing_major_object(coordinator, test_dataset):
    """
    TEST D: Same composition but high-importance anchor object (car) missing.
    Expected: Object score drops significantly due to missing high-importance element.
    """
    ref_prof = coordinator.process_reference(test_dataset["ref"])
    res_a = coordinator.evaluate_participant(test_dataset["test_a"], "P_A", ref_prof)
    res_d = coordinator.evaluate_participant(test_dataset["test_d"], "P_D", ref_prof)

    assert res_d.scores.object_accuracy < res_a.scores.object_accuracy - 5.0
    assert res_d.scores.total_score < res_a.scores.total_score - 10.0


def test_scenario_e_color_shift(coordinator, test_dataset):
    """
    TEST E: Same objects and composition but night palette / inverted colors.
    Expected: Color/lighting score drops significantly compared to Test A.
    """
    ref_prof = coordinator.process_reference(test_dataset["ref"])
    res_a = coordinator.evaluate_participant(test_dataset["test_a"], "P_A", ref_prof)
    res_e = coordinator.evaluate_participant(test_dataset["test_e"], "P_E", ref_prof)

    assert res_e.scores.color_lighting < res_a.scores.color_lighting - 4.5
    assert res_e.scores.color_lighting <= 10.5


def test_scenario_f_similar_concept_diff_scene(coordinator, test_dataset):
    """TEST F: Similar concept (desert) -> Moderate/low score (30.0 - 70.0)."""
    ref_prof = coordinator.process_reference(test_dataset["ref"])
    res = coordinator.evaluate_participant(test_dataset["test_f"], "P_F", ref_prof)

    assert 30.0 <= res.scores.total_score <= 70.0


def test_scenario_g_completely_unrelated(coordinator, test_dataset):
    """TEST G: Completely unrelated image -> Very low score (< 35.0)."""
    ref_prof = coordinator.process_reference(test_dataset["ref"])
    res = coordinator.evaluate_participant(test_dataset["test_g"], "P_G", ref_prof)

    assert res.scores.total_score < 35.0, f"Expected < 35.0 for unrelated image, got {res.scores.total_score}"


def test_scenario_h_resolution_compression_invariance(coordinator, test_dataset):
    """
    TEST H: Resized and compressed image.
    Expected: Total score remains within 5 points of Test A (resolution invariant).
    """
    ref_prof = coordinator.process_reference(test_dataset["ref"])
    res_a = coordinator.evaluate_participant(test_dataset["test_a"], "P_A", ref_prof)
    res_h = coordinator.evaluate_participant(test_dataset["test_h"], "P_H", ref_prof)

    diff = abs(res_a.scores.total_score - res_h.scores.total_score)
    assert diff <= 5.0, f"Score difference between full-res and compressed is {diff:.1f} (expected <= 5.0)"


def test_scenario_i_artistic_style_robustness(coordinator, test_dataset):
    """
    TEST I: Artistic filter / style alteration.
    Expected: Semantic similarity and object retention remain meaningful (> 60.0).
    """
    ref_prof = coordinator.process_reference(test_dataset["ref"])
    res_i = coordinator.evaluate_participant(test_dataset["test_i"], "P_I", ref_prof)

    assert res_i.scores.total_score >= 60.0
    assert res_i.scores.semantic_similarity >= 18.0


def test_scenario_j_importance_weighting(coordinator, test_dataset):
    """
    TEST J: High-importance element retention (Participant A) vs Low-importance only (Participant B).
    Expected: Participant A scores significantly higher than Participant B.
    """
    ref_prof = coordinator.process_reference(test_dataset["ref"])
    res_j1 = coordinator.evaluate_participant(test_dataset["test_j1"], "P_J1", ref_prof)
    res_j2 = coordinator.evaluate_participant(test_dataset["test_j2"], "P_J2", ref_prof)

    assert res_j1.scores.object_accuracy > res_j2.scores.object_accuracy
    assert res_j1.scores.total_score > res_j2.scores.total_score + 10.0


# ==========================================
# 2. SCORING SANITY & MONOTONIC ORDER CHECK
# ==========================================

def test_scoring_monotonic_ordering(coordinator, test_dataset):
    """
    Verify strict logical ordering across the test matrix:
    Near-identical (A) > Minor changes (B) > Moderate recreations (C, D, E) > Concept (F) > Unrelated (G)
    """
    ref_prof = coordinator.process_reference(test_dataset["ref"])
    score_a = coordinator.evaluate_participant(test_dataset["test_a"], "A", ref_prof).scores.total_score
    score_b = coordinator.evaluate_participant(test_dataset["test_b"], "B", ref_prof).scores.total_score
    score_c = coordinator.evaluate_participant(test_dataset["test_c"], "C", ref_prof).scores.total_score
    score_f = coordinator.evaluate_participant(test_dataset["test_f"], "F", ref_prof).scores.total_score
    score_g = coordinator.evaluate_participant(test_dataset["test_g"], "G", ref_prof).scores.total_score

    assert score_a >= score_b, f"A ({score_a}) should be >= B ({score_b})"
    assert score_b > score_c, f"B ({score_b}) should be > C ({score_c})"
    assert score_c > score_f, f"C ({score_c}) should be > F ({score_f})"
    assert score_f > score_g, f"F ({score_f}) should be > G ({score_g})"


# ==========================================
# 3. SCORE BOUNDS & FLOATING-POINT SAFETY
# ==========================================

def test_score_component_bounds_and_safety(coordinator, test_dataset):
    """
    Verify all 5 category scores and total score are strictly bounded:
    - Semantic: [0, 30]
    - Object: [0, 25]
    - Composition: [0, 20]
    - Color: [0, 15]
    - Details: [0, 10]
    - Total: [0, 100]
    No floating-point overflow or negative numbers.
    """
    ref_prof = coordinator.process_reference(test_dataset["ref"])
    for key in ["test_a", "test_b", "test_c", "test_d", "test_e", "test_f", "test_g"]:
        res = coordinator.evaluate_participant(test_dataset[key], key, ref_prof)
        s = res.scores
        assert 0.0 <= s.semantic_similarity <= 30.0
        assert 0.0 <= s.object_accuracy <= 25.0
        assert 0.0 <= s.composition_spatial <= 20.0
        assert 0.0 <= s.color_lighting <= 15.0
        assert 0.0 <= s.fine_details <= 10.0
        assert 0.0 <= s.total_score <= 100.0

        # Check total equals sum of components within rounding precision
        sum_cats = round(s.semantic_similarity + s.object_accuracy + s.composition_spatial + s.color_lighting + s.fine_details, 1)
        assert abs(s.total_score - sum_cats) <= 0.2


# ==========================================
# 4. EXPLANATION CONSISTENCY & CONTRADICTIONS
# ==========================================

def test_score_explanation_consistency_validator(coordinator, test_dataset):
    """
    Test automated contradiction checker on all test cases.
    Must return 0 contradictions across all evaluation results.
    """
    ref_prof = coordinator.process_reference(test_dataset["ref"])
    for key in ["test_a", "test_b", "test_c", "test_d", "test_e", "test_f", "test_g", "test_h", "test_i"]:
        res = coordinator.evaluate_participant(test_dataset[key], key, ref_prof)
        assert res.judging_report is not None
        warnings = validate_explanation_consistency(res.scores, res.judging_report)
        assert len(warnings) == 0, f"Found contradiction warnings for {key}: {warnings}"


# ==========================================
# 5. DUPLICATE IMAGE DETECTION & AUDIT
# ==========================================

def test_duplicate_image_detection(coordinator, test_dataset):
    """
    Test that submitting identical images flags them as duplicates in audit without crash.
    """
    participants = [
        ("P01", test_dataset["test_a"]),
        ("P02", test_dataset["test_b"]),
        ("P03", test_dataset["test_a"]),  # Duplicate of P01
    ]
    batch_resp = coordinator.evaluate_batch(test_dataset["ref"], participants)

    assert batch_resp.total_participants == 3
    assert batch_resp.audit_metadata is not None
    assert "P03" in batch_resp.audit_metadata.duplicate_flags
    assert batch_resp.audit_metadata.duplicate_flags["P03"] == "P01"

    p03_ranked = next(p for p in batch_resp.ranked_results if p.participant_id == "P03")
    assert p03_ranked.is_duplicate is True
    assert p03_ranked.duplicate_of == "P01"


# ==========================================
# 6. FAULT TOLERANCE & RESUME CAPABILITY
# ==========================================

def test_fault_tolerance_and_resume(coordinator, test_dataset, tmp_path):
    """
    Test corrupted image fault tolerance and resume skipping.
    """
    corrupted_path = tmp_path / "corrupted.png"
    corrupted_path.write_bytes(b"NOT_A_VALID_IMAGE_BYTES_12345")

    participants = [
        ("P01", test_dataset["test_a"]),
        ("P02_CORRUPT", corrupted_path),
        ("P03", test_dataset["test_b"]),
    ]

    # Batch should complete without crashing
    batch_resp = coordinator.evaluate_batch(test_dataset["ref"], participants)
    assert batch_resp.total_participants == 3

    corrupt_result = next(p for p in batch_resp.ranked_results if p.participant_id == "P02_CORRUPT")
    assert corrupt_result.total_score == 0.0
    assert "failed" in corrupt_result.explanation.lower()

    # Now test resume: P01 already evaluated, evaluate P04
    eval_p01 = coordinator.evaluate_participant(test_dataset["test_a"], "P01", coordinator.process_reference(test_dataset["ref"]))
    new_participants = [
        ("P01", test_dataset["test_a"]),
        ("P04", test_dataset["test_c"]),
    ]
    resume_resp = coordinator.evaluate_batch(
        test_dataset["ref"], new_participants, existing_evaluations=[eval_p01]
    )
    assert resume_resp.total_participants == 2


# ==========================================
# 7. DETERMINISTIC REPRODUCIBILITY
# ==========================================

def test_score_reproducibility(coordinator, test_dataset):
    """
    Run the same evaluation 5 times to verify 100% deterministic identical scores.
    """
    ref_prof = coordinator.process_reference(test_dataset["ref"])
    scores = []
    for _ in range(5):
        res = coordinator.evaluate_participant(test_dataset["test_b"], "P_REP", ref_prof)
        scores.append(res.scores.total_score)

    assert all(s == scores[0] for s in scores), f"Non-deterministic scores detected: {scores}"


# ==========================================
# 8. 52-PARTICIPANT DRY RUN SIMULATION
# ==========================================

def test_52_participant_competition_dry_run(coordinator, test_dataset):
    """
    Simulate the exact 52-participant competition event.
    Measures processing time, ranking correctness, and memory usage.
    """
    tracemalloc.start()
    start_time = time.time()

    # Build 52 participant list using variations
    variations = [
        test_dataset["test_a"], test_dataset["test_b"], test_dataset["test_c"],
        test_dataset["test_d"], test_dataset["test_e"], test_dataset["test_f"],
        test_dataset["test_g"], test_dataset["test_h"], test_dataset["test_i"],
    ]
    participants = []
    for i in range(1, 53):
        img_path = variations[(i - 1) % len(variations)]
        participants.append((f"P{i:02d}", img_path))

    batch_resp = coordinator.evaluate_batch(test_dataset["ref"], participants)
    elapsed = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert batch_resp.total_participants == 52
    assert len(batch_resp.ranked_results) == 52

    # Check ranking monotonicity
    for i in range(len(batch_resp.ranked_results) - 1):
        assert batch_resp.ranked_results[i].total_score >= batch_resp.ranked_results[i + 1].total_score

    # Check performance: 52 images under 30 seconds (< 0.6s per image)
    assert elapsed < 30.0, f"52-image batch took {elapsed:.2f}s (expected < 30s)"
    peak_mb = peak / (1024 * 1024)
    assert peak_mb < 300.0, f"Peak memory was {peak_mb:.1f} MB (expected < 300 MB)"


# ==========================================
# 9. 60-PARTICIPANT STRESS TEST
# ==========================================

def test_60_participant_stress_test(coordinator, test_dataset):
    """
    Maximum load stress test with 60 participants.
    Verifies system stability, audit integrity, and zero crashes.
    """
    variations = [
        test_dataset["test_a"], test_dataset["test_b"], test_dataset["test_c"],
        test_dataset["test_d"], test_dataset["test_e"], test_dataset["test_f"],
        test_dataset["test_g"], test_dataset["test_h"], test_dataset["test_i"],
    ]
    participants = [(f"P{i:02d}", variations[(i - 1) % len(variations)]) for i in range(1, 61)]

    start_time = time.time()
    batch_resp = coordinator.evaluate_batch(test_dataset["ref"], participants)
    elapsed = time.time() - start_time

    assert batch_resp.total_participants == 60
    assert len(batch_resp.ranked_results) == 60
    assert batch_resp.audit_metadata is not None
    assert batch_resp.audit_metadata.consistency_checks_passed is True
    assert elapsed < 35.0, f"60-image batch took {elapsed:.2f}s (expected < 35s)"
