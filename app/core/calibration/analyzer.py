"""
Calibration Analyzer for THE 2047 judging engine.
Performs statistical calibration between human evaluator ratings and automated AI scores.
Calculates MAE, Pearson correlation, Spearman rank correlation, ranking agreement,
category error breakdown, significant disagreement detection, and A/B weight simulation.
"""
import math
from typing import List, Dict, Tuple, Optional
import numpy as np
from scipy import stats as sp_stats

from app.models.schemas import (
    HumanEvaluation,
    EvaluationResult,
    DisagreementItem,
    CalibrationStats,
    ABConfigSimulationRequest,
    ABConfigSimulationResponse,
)
from app.config import CATEGORY_WEIGHTS, TOTAL_POINTS


class CalibrationAnalyzer:
    """
    Statistical engine for evaluating scoring calibration, human agreement,
    and simulation of alternative scoring configurations.
    """

    @staticmethod
    def compute_ranks(values: List[float], reverse: bool = True) -> List[float]:
        """
        Compute standard competition fractional ranks for tied values.
        Higher score = Rank 1 (if reverse=True).
        """
        n = len(values)
        if n == 0:
            return []
        
        # Array of (value, original_idx)
        indexed = sorted(enumerate(values), key=lambda x: x[1], reverse=reverse)
        ranks = [0.0] * n
        
        i = 0
        while i < n:
            j = i
            while j < n - 1 and indexed[j][1] == indexed[j + 1][1]:
                j += 1
            # Average rank for ties (1-based)
            avg_rank = (i + 1 + j + 1) / 2.0
            for k in range(i, j + 1):
                ranks[indexed[k][0]] = avg_rank
            i = j + 1
            
        return ranks

    @staticmethod
    def calculate_pairwise_ranking_agreement(human_scores: List[float], ai_scores: List[float]) -> float:
        """
        Calculate pairwise ranking agreement percentage.
        Percentage of participant pairs whose relative ordering agrees between human and AI.
        """
        n = len(human_scores)
        if n < 2:
            return 100.0

        concordant = 0
        discordant = 0
        ties = 0

        for i in range(n):
            for j in range(i + 1, n):
                h_diff = human_scores[i] - human_scores[j]
                a_diff = ai_scores[i] - ai_scores[j]

                if h_diff == 0 and a_diff == 0:
                    concordant += 1
                elif (h_diff > 0 and a_diff > 0) or (h_diff < 0 and a_diff < 0):
                    concordant += 1
                elif h_diff == 0 or a_diff == 0:
                    ties += 1
                    concordant += 0.5  # half credit for tie on one side
                else:
                    discordant += 1

        total_pairs = n * (n - 1) / 2
        return round((concordant / total_pairs) * 100.0, 1)

    def analyze(
        self,
        human_evals: List[HumanEvaluation],
        eval_results: List[EvaluationResult],
    ) -> CalibrationStats:
        """
        Compute full calibration metrics between Human and AI evaluations.
        """
        if not human_evals or not eval_results:
            return CalibrationStats(
                total_samples=0,
                human_avg_score=0.0,
                ai_avg_score=0.0,
                mean_absolute_error=0.0,
                mean_difference=0.0,
                pearson_correlation=0.0,
                spearman_rank_correlation=0.0,
                ranking_agreement_pct=0.0,
                category_mae={},
                significant_disagreements=[],
                decision="🟡 CALIBRATION NEEDS IMPROVEMENT",
                decision_rationale="Insufficient samples provided for calibration analysis.",
                recommendations=["Upload at least 5 sample participant evaluations to perform calibration."],
            )

        ai_map = {e.participant_id: e for e in eval_results}
        matched_human: List[float] = []
        matched_ai: List[float] = []
        disagreements: List[DisagreementItem] = []

        cat_diffs: Dict[str, List[float]] = {
            "semantic_similarity": [],
            "object_accuracy": [],
            "composition_spatial": [],
            "color_lighting": [],
            "fine_details": [],
        }

        for h in human_evals:
            if h.participant_id not in ai_map:
                continue
            ev = ai_map[h.participant_id]
            h_score = round(float(h.human_score), 1)
            ai_score = round(float(ev.scores.total_score), 1)

            matched_human.append(h_score)
            matched_ai.append(ai_score)

            diff = round(h_score - ai_score, 1)
            abs_diff = round(abs(diff), 1)

            # Category difference inspection
            cat_deltas: Dict[str, float] = {}
            largest_cat = "Overall"
            max_norm_err = 0.0

            if h.human_semantic is not None:
                d_sem = round(abs(h.human_semantic - ev.scores.semantic_similarity), 1)
                cat_diffs["semantic_similarity"].append(d_sem)
                cat_deltas["semantic"] = round(h.human_semantic - ev.scores.semantic_similarity, 1)
                if (d_sem / 30.0) > max_norm_err:
                    max_norm_err = d_sem / 30.0
                    largest_cat = "Semantic Similarity"

            if h.human_object is not None:
                d_obj = round(abs(h.human_object - ev.scores.object_accuracy), 1)
                cat_diffs["object_accuracy"].append(d_obj)
                cat_deltas["object"] = round(h.human_object - ev.scores.object_accuracy, 1)
                if (d_obj / 25.0) > max_norm_err:
                    max_norm_err = d_obj / 25.0
                    largest_cat = "Object Accuracy"

            if h.human_composition is not None:
                d_comp = round(abs(h.human_composition - ev.scores.composition_spatial), 1)
                cat_diffs["composition_spatial"].append(d_comp)
                cat_deltas["composition"] = round(h.human_composition - ev.scores.composition_spatial, 1)
                if (d_comp / 20.0) > max_norm_err:
                    max_norm_err = d_comp / 20.0
                    largest_cat = "Composition"

            if h.human_color is not None:
                d_col = round(abs(h.human_color - ev.scores.color_lighting), 1)
                cat_diffs["color_lighting"].append(d_col)
                cat_deltas["color"] = round(h.human_color - ev.scores.color_lighting, 1)
                if (d_col / 15.0) > max_norm_err:
                    max_norm_err = d_col / 15.0
                    largest_cat = "Color & Lighting"

            if h.human_detail is not None:
                d_det = round(abs(h.human_detail - ev.scores.fine_details), 1)
                cat_diffs["fine_details"].append(d_det)
                cat_deltas["details"] = round(h.human_detail - ev.scores.fine_details, 1)
                if (d_det / 10.0) > max_norm_err:
                    max_norm_err = d_det / 10.0
                    largest_cat = "Fine Details"

            is_sig = abs_diff > 10.0
            disagreements.append(
                DisagreementItem(
                    participant_id=h.participant_id,
                    human_score=h_score,
                    ai_score=ai_score,
                    difference=diff,
                    abs_difference=abs_diff,
                    is_significant=is_sig,
                    main_category_disagreement=largest_cat,
                    category_differences=cat_deltas,
                    human_comments=h.human_comments,
                )
            )

        n = len(matched_human)
        if n == 0:
            return CalibrationStats(
                total_samples=0,
                human_avg_score=0.0,
                ai_avg_score=0.0,
                mean_absolute_error=0.0,
                mean_difference=0.0,
                pearson_correlation=0.0,
                spearman_rank_correlation=0.0,
                ranking_agreement_pct=0.0,
                category_mae={},
                significant_disagreements=[],
                decision="🟡 CALIBRATION NEEDS IMPROVEMENT",
                decision_rationale="None of the submitted human ratings matched available AI evaluations.",
                recommendations=["Ensure participant IDs match between human evaluations and evaluated images."],
            )

        human_avg = round(float(np.mean(matched_human)), 1)
        ai_avg = round(float(np.mean(matched_ai)), 1)
        mae = round(float(np.mean([abs(h - a) for h, a in zip(matched_human, matched_ai)])), 2)
        mean_diff = round(float(np.mean([h - a for h, a in zip(matched_human, matched_ai)])), 2)

        # Pearson correlation
        if n > 1 and np.std(matched_human) > 1e-6 and np.std(matched_ai) > 1e-6:
            pearson_r, _ = sp_stats.pearsonr(matched_human, matched_ai)
            pearson_val = round(float(pearson_r), 3) if not math.isnan(pearson_r) else 0.0
        else:
            pearson_val = 1.0 if np.allclose(matched_human, matched_ai) else 0.0

        # Spearman rank correlation
        if n > 1 and np.std(matched_human) > 1e-6 and np.std(matched_ai) > 1e-6:
            spearman_rho, _ = sp_stats.spearmanr(matched_human, matched_ai)
            spearman_val = round(float(spearman_rho), 3) if not math.isnan(spearman_rho) else 0.0
        else:
            spearman_val = 1.0 if np.allclose(matched_human, matched_ai) else 0.0

        # Pairwise ranking agreement
        rank_agreement = self.calculate_pairwise_ranking_agreement(matched_human, matched_ai)

        # Category MAEs
        category_mae: Dict[str, float] = {}
        for cat, diffs in cat_diffs.items():
            if diffs:
                category_mae[cat] = round(float(np.mean(diffs)), 2)

        # Significant disagreements sorted by absolute difference descending
        sig_disagreements = sorted(
            [d for d in disagreements if d.is_significant],
            key=lambda x: x.abs_difference,
            reverse=True,
        )

        # Automated Calibration Decision Logic
        recommendations = []
        if n < 5:
            decision = "🟡 CALIBRATION NEEDS IMPROVEMENT"
            rationale = (
                f"Calibration dataset size (N={n}) is too small for definitive validation. "
                f"Observed MAE: {mae:.2f}, Pearson r: {pearson_val:.3f}. Recommended sample size is at least 7-10."
            )
            recommendations.append("Upload additional sample participant recreations spanning high to low fidelity.")
        elif mae <= 8.5 and pearson_val >= 0.82 and rank_agreement >= 75.0:
            decision = "🟢 CALIBRATION ACCEPTABLE"
            rationale = (
                f"High human-AI alignment: MAE is {mae:.2f} pts (<= 8.5 pts threshold), "
                f"Pearson correlation is {pearson_val:.3f} (>= 0.82), and Ranking Agreement is {rank_agreement:.1f}%. "
                f"Significant disagreements ({len(sig_disagreements)}/{n}) are within expected variance for visual memory judging."
            )
            if mean_diff > 3.0:
                recommendations.append("AI shows mild conservative scoring bias (scoring ~" + f"{mean_diff:.1f}" + " pts lower than humans).")
            elif mean_diff < -3.0:
                recommendations.append("AI shows mild generous scoring bias (scoring ~" + f"{abs(mean_diff):.1f}" + " pts higher than humans).")
        elif mae <= 14.0 and pearson_val >= 0.65:
            decision = "🟡 CALIBRATION NEEDS IMPROVEMENT"
            rationale = (
                f"Moderate alignment: MAE is {mae:.2f} pts, Pearson r: {pearson_val:.3f}, Ranking Agreement: {rank_agreement:.1f}%. "
                f"There are {len(sig_disagreements)} significant disagreements (>10 pts diff)."
            )
            if category_mae:
                worst_cat = max(category_mae.items(), key=lambda x: x[1])
                recommendations.append(f"Highest category divergence observed in '{worst_cat[0]}' (MAE: {worst_cat[1]:.2f}). Inspect feature weights.")
        else:
            decision = "🔴 SCORING SYSTEM REQUIRES MAJOR REVISION"
            rationale = (
                f"Poor alignment: MAE of {mae:.2f} pts exceeds acceptable threshold, "
                f"or Pearson correlation ({pearson_val:.3f}) / Ranking Agreement ({rank_agreement:.1f}%) indicate significant divergence."
            )
            recommendations.append("Perform systematic audit of feature extractors and element importance weighting.")

        if n < 15:
            recommendations.append("Notice: Calibration dataset is small (N=" + str(n) + "). Avoid overfitting category weights to test set.")

        return CalibrationStats(
            total_samples=n,
            human_avg_score=human_avg,
            ai_avg_score=ai_avg,
            mean_absolute_error=mae,
            mean_difference=mean_diff,
            pearson_correlation=pearson_val,
            spearman_rank_correlation=spearman_val,
            ranking_agreement_pct=rank_agreement,
            category_mae=category_mae,
            significant_disagreements=sig_disagreements,
            decision=decision,
            decision_rationale=rationale,
            recommendations=recommendations,
        )

    def simulate_ab_weights(
        self,
        request: ABConfigSimulationRequest,
        eval_results: List[EvaluationResult],
    ) -> ABConfigSimulationResponse:
        """
        Simulate alternative scoring category weights against human evaluations.
        Compares Config A (30/25/20/15/10) vs Proposed Config B.
        """
        human_evals = request.human_evaluations
        ai_map = {e.participant_id: e for e in eval_results}

        # Config A weights (Default)
        w_a = {
            "semantic": 30.0,
            "object": 25.0,
            "composition": 20.0,
            "color": 15.0,
            "detail": 10.0,
        }

        # Config B weights (Normalized to sum to 100.0)
        raw_b_sum = (
            request.semantic_weight
            + request.object_weight
            + request.composition_weight
            + request.color_weight
            + request.detail_weight
        )
        if raw_b_sum <= 0:
            scale_b = 1.0
        else:
            scale_b = 100.0 / raw_b_sum

        w_b = {
            "semantic": round(request.semantic_weight * scale_b, 1),
            "object": round(request.object_weight * scale_b, 1),
            "composition": round(request.composition_weight * scale_b, 1),
            "color": round(request.color_weight * scale_b, 1),
            "detail": round(request.detail_weight * scale_b, 1),
        }

        matched_human: List[float] = []
        scores_a: List[float] = []
        scores_b: List[float] = []

        for h in human_evals:
            if h.participant_id not in ai_map:
                continue
            ev = ai_map[h.participant_id]
            matched_human.append(float(h.human_score))

            # Raw normalized metrics [0, 1]
            m = ev.comparison_metrics
            if m:
                sem_raw = m.semantic_similarity_raw
                obj_raw = m.weighted_object_score_raw
                comp_raw = m.composition_score_raw
                col_raw = m.color_lighting_score_raw
                det_raw = m.fine_detail_score_raw
            else:
                # Fallback to current category scores normalized by default weights
                sem_raw = ev.scores.semantic_similarity / 30.0
                obj_raw = ev.scores.object_accuracy / 25.0
                comp_raw = ev.scores.composition_spatial / 20.0
                col_raw = ev.scores.color_lighting / 15.0
                det_raw = ev.scores.fine_details / 10.0

            # Score A
            s_a = round(
                sem_raw * w_a["semantic"]
                + obj_raw * w_a["object"]
                + comp_raw * w_a["composition"]
                + col_raw * w_a["color"]
                + det_raw * w_a["detail"],
                1,
            )
            scores_a.append(min(100.0, max(0.0, s_a)))

            # Score B
            s_b = round(
                sem_raw * w_b["semantic"]
                + obj_raw * w_b["object"]
                + comp_raw * w_b["composition"]
                + col_raw * w_b["color"]
                + det_raw * w_b["detail"],
                1,
            )
            scores_b.append(min(100.0, max(0.0, s_b)))

        n = len(matched_human)
        if n == 0:
            return ABConfigSimulationResponse(
                config_a_name="Default (30/25/20/15/10)",
                config_a_mae=0.0,
                config_a_correlation=0.0,
                config_a_ranking_agreement_pct=0.0,
                config_b_name=request.config_name,
                config_b_weights=w_b,
                config_b_mae=0.0,
                config_b_correlation=0.0,
                config_b_ranking_agreement_pct=0.0,
                mae_improvement=0.0,
                recommendation="No matching human evaluations found for A/B simulation.",
            )

        mae_a = round(float(np.mean([abs(h - a) for h, a in zip(matched_human, scores_a)])), 2)
        mae_b = round(float(np.mean([abs(h - b) for h, b in zip(matched_human, scores_b)])), 2)

        # Correlations
        corr_a = 0.0
        corr_b = 0.0
        if n > 1:
            if np.std(matched_human) > 1e-6 and np.std(scores_a) > 1e-6:
                r_a, _ = sp_stats.pearsonr(matched_human, scores_a)
                corr_a = round(float(r_a), 3) if not math.isnan(r_a) else 0.0
            if np.std(matched_human) > 1e-6 and np.std(scores_b) > 1e-6:
                r_b, _ = sp_stats.pearsonr(matched_human, scores_b)
                corr_b = round(float(r_b), 3) if not math.isnan(r_b) else 0.0

        rank_a = self.calculate_pairwise_ranking_agreement(matched_human, scores_a)
        rank_b = self.calculate_pairwise_ranking_agreement(matched_human, scores_b)

        improvement = round(mae_a - mae_b, 2)

        if n < 10:
            rec = f"Sample size (N={n}) is too small to justify changing default weights without risk of overfitting. Retain Default (30/25/20/15/10)."
        elif improvement > 1.5 and rank_b >= rank_a:
            rec = f"Proposed configuration reduces MAE by {improvement:.2f} pts with equal or superior ranking agreement. Consider for organizer review."
        elif improvement > 0.0:
            rec = f"Proposed configuration shows marginal MAE change ({improvement:+.2f} pts). Default configuration is recommended to avoid overfitting."
        else:
            rec = f"Proposed configuration worsens MAE by {abs(improvement):.2f} pts. Retain Default configuration."

        return ABConfigSimulationResponse(
            config_a_name="Default (30/25/20/15/10)",
            config_a_mae=mae_a,
            config_a_correlation=corr_a,
            config_a_ranking_agreement_pct=rank_a,
            config_b_name=request.config_name,
            config_b_weights=w_b,
            config_b_mae=mae_b,
            config_b_correlation=corr_b,
            config_b_ranking_agreement_pct=rank_b,
            mae_improvement=improvement,
            recommendation=rec,
        )
