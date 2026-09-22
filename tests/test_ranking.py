"""
Unit tests for the StandardRankingEngine and tie-handling logic.
"""
from app.models.schemas import EvaluationResult, CategoryScores, StructuredEvidence
from app.core.ranking.engine import StandardRankingEngine


def test_ranking_ties_handling():
    engine = StandardRankingEngine()

    scores_p1 = CategoryScores(
        semantic_similarity=27.0,
        object_accuracy=23.0,
        composition_spatial=18.0,
        color_lighting=14.0,
        fine_details=9.0,
        total_score=91.0,
    )

    scores_p2 = CategoryScores(
        semantic_similarity=26.0,
        object_accuracy=24.0,
        composition_spatial=17.0,
        color_lighting=15.0,
        fine_details=9.0,
        total_score=91.0,  # Exact tie with P1
    )

    scores_p3 = CategoryScores(
        semantic_similarity=20.0,
        object_accuracy=18.0,
        composition_spatial=15.0,
        color_lighting=10.0,
        fine_details=7.0,
        total_score=70.0,
    )

    scores_p4 = CategoryScores(
        semantic_similarity=10.0,
        object_accuracy=10.0,
        composition_spatial=10.0,
        color_lighting=5.0,
        fine_details=3.0,
        total_score=38.0,
    )

    evaluations = [
        EvaluationResult(participant_id="P-01", filename="p1.png", scores=scores_p1),
        EvaluationResult(participant_id="P-02", filename="p2.png", scores=scores_p2),
        EvaluationResult(participant_id="P-03", filename="p3.png", scores=scores_p3),
        EvaluationResult(participant_id="P-04", filename="p4.png", scores=scores_p4),
    ]

    ranked = engine.rank(evaluations)

    assert len(ranked) == 4
    # P1 and P2 must both be Rank 1 and flagged is_tied=True
    assert ranked[0].rank == 1
    assert ranked[1].rank == 1
    assert ranked[0].is_tied is True
    assert ranked[1].is_tied is True
    assert ranked[0].total_score == 91.0
    assert ranked[1].total_score == 91.0

    # P3 must be Rank 3 (since two participants tied for Rank 1)
    assert ranked[2].rank == 3
    assert ranked[2].is_tied is False
    assert ranked[2].total_score == 70.0

    # P4 must be Rank 4
    assert ranked[3].rank == 4
    assert ranked[3].is_tied is False
    assert ranked[3].total_score == 38.0
