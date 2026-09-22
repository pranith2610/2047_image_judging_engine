"""
Standard ranking engine implementation.
Performs deterministic competition ranking with proper tie handling and percentile calculation.
"""
from typing import List
from app.models.schemas import EvaluationResult, RankedParticipant
from app.core.ranking.base import BaseRankingEngine


class StandardRankingEngine(BaseRankingEngine):
    """
    Ranks competition participants deterministically by final score.
    Properly handles ties: identical final scores share the same rank
    and are flagged as tied without arbitrary tie-breaking.
    """

    def rank(self, evaluations: List[EvaluationResult]) -> List[RankedParticipant]:
        if not evaluations:
            return []

        # Sort purely by total_score descending (with stable participant_id for deterministic order)
        sorted_evals = sorted(
            evaluations,
            key=lambda e: (-e.scores.total_score, e.participant_id)
        )

        n = len(sorted_evals)
        ranked: List[RankedParticipant] = []

        # Count frequencies of scores to detect ties
        score_counts = {}
        for ev in sorted_evals:
            s = ev.scores.total_score
            score_counts[s] = score_counts.get(s, 0) + 1

        for i, ev in enumerate(sorted_evals):
            score = ev.scores.total_score
            is_tied = score_counts[score] > 1

            if i > 0 and score == sorted_evals[i - 1].scores.total_score:
                rank_num = ranked[i - 1].rank
            else:
                rank_num = i + 1

            # Percentile: (N - rank + 1) / N * 100
            percentile = round(((n - rank_num + 1) / n) * 100.0, 1)

            # Competition tier
            if rank_num == 1:
                tier = "Grand Champion"
            elif rank_num in (2, 3):
                tier = "Podium Finalist"
            elif percentile >= 75:
                tier = "Distinguished"
            elif percentile >= 50:
                tier = "Proficient"
            else:
                tier = "Contender"

            # Use score_tier from judging_report if present for match summary, or keep percentile tier
            match_tier = ev.judging_report.score_tier if ev.judging_report else tier

            ranked.append(
                RankedParticipant(
                    rank=rank_num,
                    participant_id=ev.participant_id,
                    filename=ev.filename,
                    total_score=ev.scores.total_score,
                    scores=ev.scores,
                    explanation=ev.explanation,
                    tier=match_tier,
                    percentile=percentile,
                    is_tied=is_tied,
                    is_duplicate=ev.is_duplicate,
                    duplicate_of=ev.duplicate_of,
                    image_hash=ev.image_hash,
                    evidence=ev.evidence,
                    judging_report=ev.judging_report,
                    human_evaluation=ev.human_evaluation,
                    human_override=ev.human_override,
                    flagged_for_review=ev.flagged_for_review,
                )
            )

        return ranked
