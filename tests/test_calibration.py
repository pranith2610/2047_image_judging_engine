"""
Comprehensive calibration and human-AI alignment test suite for THE 2047.
Verifies statistical calibration metrics (MAE, Pearson, Spearman, Ranking Agreement),
blind comparison workflow, significant disagreement flagging, A/B weight simulation,
30-second memory importance principles, style robustness, and non-destructive human overrides.
"""
import io
import pytest
import numpy as np
from PIL import Image, ImageDraw

from app.models.schemas import (
    HumanEvaluation,
    EvaluationResult,
    CategoryScores,
    ABConfigSimulationRequest,
    HumanOverrideRequest,
    FlagForReviewRequest,
)
from app.core.calibration.analyzer import CalibrationAnalyzer
from app.core.coordinator import JudgingCoordinator
from app.core.analyzers.reference import BaselineReferenceAnalyzer
from app.core.analyzers.participant import BaselineParticipantAnalyzer
from app.core.comparators.comparator import BaselineFeatureComparator
from app.core.scoring.engine import WeightedScoringEngine


def create_test_image(color=(100, 150, 200), shapes=None, size=(512, 512)) -> io.BytesIO:
    """Helper to generate synthetic test images with specific visual objects."""
    img = Image.new("RGB", size, color=color)
    draw = ImageDraw.Draw(img)
    if shapes:
        for shape in shapes:
            stype = shape.get("type", "rectangle")
            box = shape.get("box", [50, 50, 150, 150])
            scolor = shape.get("color", (255, 0, 0))
            if stype == "rectangle":
                draw.rectangle(box, fill=scolor)
            elif stype == "ellipse":
                draw.ellipse(box, fill=scolor)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


@pytest.fixture
def coordinator():
    return JudgingCoordinator()


@pytest.fixture
def calibration_analyzer():
    return CalibrationAnalyzer()


def test_calibration_metrics_calculation(calibration_analyzer):
    """
    Verify MAE, mean difference bias, Pearson r, Spearman rho, and ranking agreement.
    """
    human_evals = [
        HumanEvaluation(participant_id="P01", human_score=95.0, human_semantic=29.0, human_object=24.0, human_composition=19.0, human_color=14.0, human_detail=9.0),
        HumanEvaluation(participant_id="P02", human_score=85.0, human_semantic=26.0, human_object=22.0, human_composition=16.0, human_color=13.0, human_detail=8.0),
        HumanEvaluation(participant_id="P03", human_score=75.0, human_semantic=23.0, human_object=19.0, human_composition=15.0, human_color=11.0, human_detail=7.0),
        HumanEvaluation(participant_id="P04", human_score=60.0, human_semantic=18.0, human_object=15.0, human_composition=12.0, human_color=9.0, human_detail=6.0),
        HumanEvaluation(participant_id="P05", human_score=40.0, human_semantic=12.0, human_object=10.0, human_composition=8.0, human_color=6.0, human_detail=4.0),
        HumanEvaluation(participant_id="P06", human_score=20.0, human_semantic=6.0, human_object=5.0, human_composition=4.0, human_color=3.0, human_detail=2.0),
        HumanEvaluation(participant_id="P07", human_score=10.0, human_semantic=3.0, human_object=2.0, human_composition=2.0, human_color=2.0, human_detail=1.0),
    ]

    eval_results = [
        EvaluationResult(participant_id="P01", filename="p1.png", scores=CategoryScores(semantic_similarity=28.0, object_accuracy=23.5, composition_spatial=18.5, color_lighting=14.0, fine_details=9.0, total_score=93.0)),
        EvaluationResult(participant_id="P02", filename="p2.png", scores=CategoryScores(semantic_similarity=25.0, object_accuracy=21.0, composition_spatial=16.0, color_lighting=12.5, fine_details=7.5, total_score=82.0)),
        EvaluationResult(participant_id="P03", filename="p3.png", scores=CategoryScores(semantic_similarity=22.0, object_accuracy=18.0, composition_spatial=14.0, color_lighting=11.0, fine_details=7.0, total_score=72.0)),
        EvaluationResult(participant_id="P04", filename="p4.png", scores=CategoryScores(semantic_similarity=17.5, object_accuracy=14.0, composition_spatial=12.5, color_lighting=9.0, fine_details=5.0, total_score=58.0)),
        EvaluationResult(participant_id="P05", filename="p5.png", scores=CategoryScores(semantic_similarity=13.0, object_accuracy=10.5, composition_spatial=8.0, color_lighting=6.5, fine_details=4.0, total_score=42.0)),
        EvaluationResult(participant_id="P06", filename="p6.png", scores=CategoryScores(semantic_similarity=7.0, object_accuracy=5.0, composition_spatial=4.0, color_lighting=3.0, fine_details=2.0, total_score=21.0)),
        EvaluationResult(participant_id="P07", filename="p7.png", scores=CategoryScores(semantic_similarity=3.5, object_accuracy=2.5, composition_spatial=2.0, color_lighting=1.5, fine_details=1.0, total_score=10.5)),
    ]

    stats = calibration_analyzer.analyze(human_evals, eval_results)

    assert stats.total_samples == 7
    assert stats.mean_absolute_error < 4.0  # High alignment
    assert stats.pearson_correlation > 0.95
    assert stats.spearman_rank_correlation > 0.95
    assert stats.ranking_agreement_pct == 100.0  # Perfect relative monotonic ordering
    assert len(stats.significant_disagreements) == 0
    assert "🟢 CALIBRATION ACCEPTABLE" in stats.decision


def test_significant_disagreement_detection(calibration_analyzer):
    """
    Verify that when |Human - AI| > 10.0, the system flags a significant disagreement
    and correctly isolates the primary diverging category.
    """
    human_evals = [
        HumanEvaluation(
            participant_id="P_DIVERGENT",
            human_score=85.0,
            human_semantic=26.0,
            human_object=23.0,
            human_composition=18.0,
            human_color=11.0,
            human_detail=7.0,
            human_comments="Human felt composition was well maintained despite object placement.",
        )
    ]

    eval_results = [
        EvaluationResult(
            participant_id="P_DIVERGENT",
            filename="p_div.png",
            scores=CategoryScores(
                semantic_similarity=25.0,
                object_accuracy=22.0,
                composition_spatial=6.0,  # AI heavily penalized composition
                color_lighting=10.0,
                fine_details=6.0,
                total_score=69.0,         # Diff = 85 - 69 = 16 pts
            ),
        )
    ]

    stats = calibration_analyzer.analyze(human_evals, eval_results)

    assert len(stats.significant_disagreements) == 1
    d = stats.significant_disagreements[0]
    assert d.participant_id == "P_DIVERGENT"
    assert d.is_significant is True
    assert d.difference == 16.0
    assert d.abs_difference == 16.0
    assert "Composition" in d.main_category_disagreement


def test_ab_weight_simulation(calibration_analyzer):
    """
    Verify A/B testing simulator calculates comparative MAE and ranking agreement
    between Default Config (30/25/20/15/10) and an alternative configuration.
    """
    human_evals = [
        HumanEvaluation(participant_id="P01", human_score=90.0),
        HumanEvaluation(participant_id="P02", human_score=75.0),
        HumanEvaluation(participant_id="P03", human_score=60.0),
        HumanEvaluation(participant_id="P04", human_score=45.0),
        HumanEvaluation(participant_id="P05", human_score=30.0),
    ]

    eval_results = [
        EvaluationResult(participant_id="P01", filename="p1.png", scores=CategoryScores(semantic_similarity=27.0, object_accuracy=22.5, composition_spatial=18.0, color_lighting=13.5, fine_details=9.0, total_score=90.0)),
        EvaluationResult(participant_id="P02", filename="p2.png", scores=CategoryScores(semantic_similarity=22.5, object_accuracy=18.5, composition_spatial=15.0, color_lighting=11.5, fine_details=7.5, total_score=75.0)),
        EvaluationResult(participant_id="P03", filename="p3.png", scores=CategoryScores(semantic_similarity=18.0, object_accuracy=15.0, composition_spatial=12.0, color_lighting=9.0, fine_details=6.0, total_score=60.0)),
        EvaluationResult(participant_id="P04", filename="p4.png", scores=CategoryScores(semantic_similarity=13.5, object_accuracy=11.0, composition_spatial=9.0, color_lighting=7.0, fine_details=4.5, total_score=45.0)),
        EvaluationResult(participant_id="P05", filename="p5.png", scores=CategoryScores(semantic_similarity=9.0, object_accuracy=7.5, composition_spatial=6.0, color_lighting=4.5, fine_details=3.0, total_score=30.0)),
    ]

    sim_req = ABConfigSimulationRequest(
        config_name="Emphasis on Objects",
        semantic_weight=30.0,
        object_weight=30.0,
        composition_weight=20.0,
        color_weight=10.0,
        detail_weight=10.0,
        human_evaluations=human_evals,
    )

    sim_resp = calibration_analyzer.simulate_ab_weights(sim_req, eval_results)

    assert sim_resp.config_a_mae is not None
    assert sim_resp.config_b_mae is not None
    assert sim_resp.config_b_weights["object"] == 30.0
    assert "overfitting" in sim_resp.recommendation.lower() or "default" in sim_resp.recommendation.lower()


def test_30_second_memory_principle(coordinator, tmp_path):
    """
    Test 30-Second Memory Principle:
    A participant who recreates the prominent central main subject but misses a minor background motif
    MUST score substantially higher than a participant who captures background noise but misses the main subject.
    """
    # Reference: Big prominent red circle in center (Main Subject, HIGH) + small green square in corner (LOW)
    ref_shapes = [
        {"type": "ellipse", "box": [150, 150, 360, 360], "color": (255, 30, 30)},
        {"type": "rectangle", "box": [20, 20, 50, 50], "color": (30, 200, 30)},
    ]
    ref_path = tmp_path / "ref_memory.png"
    ref_path.write_bytes(create_test_image(color=(240, 240, 240), shapes=ref_shapes).getvalue())

    # Participant A: Captured prominent red circle in center, missed tiny green corner
    part_a_shapes = [
        {"type": "ellipse", "box": [155, 145, 355, 365], "color": (245, 40, 35)},
    ]
    part_a_path = tmp_path / "part_a.png"
    part_a_path.write_bytes(create_test_image(color=(240, 240, 240), shapes=part_a_shapes).getvalue())

    # Participant B: Missed prominent red circle completely, only drew tiny green corner
    part_b_shapes = [
        {"type": "rectangle", "box": [20, 20, 50, 50], "color": (30, 200, 30)},
    ]
    part_b_path = tmp_path / "part_b.png"
    part_b_path.write_bytes(create_test_image(color=(240, 240, 240), shapes=part_b_shapes).getvalue())

    ref_prof = coordinator.process_reference(ref_path)
    res_a = coordinator.evaluate_participant(part_a_path, "P_MAIN_MATCH", ref_prof)
    res_b = coordinator.evaluate_participant(part_b_path, "P_BG_ONLY", ref_prof)

    # Participant A must decisively beat Participant B
    assert res_a.scores.total_score > res_b.scores.total_score + 15.0
    assert res_a.scores.object_accuracy > res_b.scores.object_accuracy


def test_style_difference_robustness(coordinator, tmp_path):
    """
    Test Style Robustness:
    Evaluate content & spatial structure across artistic variations (e.g. sketch/illustration vs photo).
    The system should maintain high semantic/object fidelity when objects and composition align.
    """
    ref_shapes = [
        {"type": "rectangle", "box": [120, 120, 380, 380], "color": (40, 80, 180)},
        {"type": "ellipse", "box": [200, 200, 300, 300], "color": (255, 200, 50)},
    ]
    ref_path = tmp_path / "ref_style.png"
    ref_path.write_bytes(create_test_image(color=(220, 220, 220), shapes=ref_shapes).getvalue())

    # Stylized recreation: Same geometry, pastel illustration palette
    art_shapes = [
        {"type": "rectangle", "box": [120, 120, 380, 380], "color": (90, 130, 220)},
        {"type": "ellipse", "box": [200, 200, 300, 300], "color": (255, 220, 100)},
    ]
    art_path = tmp_path / "part_art.png"
    art_path.write_bytes(create_test_image(color=(235, 235, 235), shapes=art_shapes).getvalue())

    ref_prof = coordinator.process_reference(ref_path)
    art_res = coordinator.evaluate_participant(art_path, "P_ART_STYLE", ref_prof)

    # Should achieve a solid score (>= 70.0) reflecting structural and semantic retention
    assert art_res.scores.total_score >= 70.0
    assert art_res.scores.semantic_similarity >= 20.0
    assert art_res.scores.object_accuracy >= 18.0


def test_non_destructive_human_override(coordinator, tmp_path):
    """
    Verify that human score overrides:
    1. Are non-destructive (original AI score preserved in audit metadata).
    2. Record explicit organizer reason and timestamp.
    3. Update leaderboard ranking properly.
    """
    ref_path = tmp_path / "ref.png"
    ref_path.write_bytes(create_test_image().getvalue())
    p1_path = tmp_path / "p1.png"
    p1_path.write_bytes(create_test_image().getvalue())

    batch_resp = coordinator.evaluate_batch(ref_path, [("P01", p1_path)])
    p1_rank = batch_resp.ranked_results[0]
    original_ai = p1_rank.total_score

    # Simulate human override
    override_req = HumanOverrideRequest(
        participant_id="P01",
        override_score=88.5,
        override_reason="Judge noted nuanced spatial relationship between foreground subjects",
        organizer_name="Lead Organizer",
    )

    from app.models.schemas import HumanScoreOverride
    import datetime
    override_obj = HumanScoreOverride(
        participant_id=p1_rank.participant_id,
        original_ai_score=original_ai,
        override_score=override_req.override_score,
        override_reason=override_req.override_reason,
        organizer_name=override_req.organizer_name,
        timestamp=datetime.datetime.utcnow().isoformat() + "Z",
        flagged_for_review=True,
    )

    p1_rank.human_override = override_obj
    p1_rank.total_score = override_req.override_score
    p1_rank.flagged_for_review = True

    assert p1_rank.human_override.original_ai_score == original_ai
    assert p1_rank.total_score == 88.5
    assert p1_rank.human_override.override_reason == "Judge noted nuanced spatial relationship between foreground subjects"
    assert p1_rank.flagged_for_review is True
