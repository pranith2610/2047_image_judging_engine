"""
Weighted scoring engine implementation.
Applies category weights and element importance to produce deterministic 100-point scores.
"""
from app.config import CATEGORY_WEIGHTS, TOTAL_POINTS
from app.models.schemas import (
    FeatureComparisonMetrics,
    ReferenceProfile,
    CategoryScores,
)
from app.core.scoring.base import BaseScoringEngine


class WeightedScoringEngine(BaseScoringEngine):
    """
    Official weighted competition scoring engine for THE 2047.
    Enforces strict mathematical boundaries:
    - Semantic: 30.0 pts
    - Object: 25.0 pts
    - Composition: 20.0 pts
    - Color/Lighting: 15.0 pts
    - Fine Details: 10.0 pts
    Total: 100.0 pts
    """

    def calculate_scores(
        self, metrics: FeatureComparisonMetrics, reference: ReferenceProfile
    ) -> CategoryScores:
        # Category 1: Semantic / Overall Visual Similarity (Max 30.0)
        max_sem = float(CATEGORY_WEIGHTS.get("semantic_similarity", 30.0))
        semantic_score = round(
            float(min(max_sem, max(0.0, metrics.semantic_similarity_raw * max_sem))), 1
        )

        # Category 2: Object / Element Accuracy (Max 25.0)
        max_obj = float(CATEGORY_WEIGHTS.get("object_accuracy", 25.0))
        raw_obj_pts = metrics.weighted_object_score_raw * max_obj

        # Anti-False-Positive Rule: If HIGH importance elements were missed, apply proportional penalty
        if metrics.high_importance_found_ratio < 0.25:
            # Severe major-feature absence penalty
            raw_obj_pts = min(2.0, raw_obj_pts)
        elif metrics.high_importance_found_ratio < 1.0:
            missed_penalty = (1.0 - metrics.high_importance_found_ratio) * 6.0
            raw_obj_pts = max(0.0, raw_obj_pts - missed_penalty)

        object_score = round(float(min(max_obj, max(0.0, raw_obj_pts))), 1)

        # Category 3: Composition & Spatial Arrangement (Max 20.0)
        max_comp = float(CATEGORY_WEIGHTS.get("composition_spatial", 20.0))
        comp_pts = metrics.composition_score_raw * max_comp
        if metrics.high_importance_found_ratio < 0.25:
            comp_pts = min(2.5, comp_pts)
        composition_score = round(
            float(min(max_comp, max(0.0, comp_pts))), 1
        )

        # Category 4: Color & Lighting (Max 15.0)
        max_color = float(CATEGORY_WEIGHTS.get("color_lighting", 15.0))
        color_pts = metrics.color_lighting_score_raw * max_color
        if metrics.high_importance_found_ratio < 0.25 and metrics.semantic_similarity_raw < 0.25:
            color_pts = min(3.0, color_pts)
        color_score = round(
            float(min(max_color, max(0.0, color_pts))), 1
        )

        # Category 5: Fine Details (Max 10.0)
        max_det = float(CATEGORY_WEIGHTS.get("fine_details", 10.0))
        det_pts = metrics.fine_detail_score_raw * max_det
        if metrics.high_importance_found_ratio < 0.25:
            det_pts = min(1.5, det_pts)
        detail_score = round(
            float(min(max_det, max(0.0, det_pts))), 1
        )

        # Scene Identity Gate: Check for fundamental scene/subject mismatch
        raw_total = semantic_score + object_score + composition_score + color_score + detail_score

        if metrics.high_importance_found_ratio < 0.25 and metrics.semantic_similarity_raw < 0.25:
            ceiling = 14.5
            if raw_total > ceiling:
                scale = ceiling / max(0.1, raw_total)
                semantic_score = round(semantic_score * scale, 1)
                object_score = round(object_score * scale, 1)
                composition_score = round(composition_score * scale, 1)
                color_score = round(color_score * scale, 1)
                detail_score = round(detail_score * scale, 1)
                raw_total = round(semantic_score + object_score + composition_score + color_score + detail_score, 1)

        total = round(
            float(min(TOTAL_POINTS, max(0.0, raw_total))),
            1,
        )

        return CategoryScores(
            semantic_similarity=semantic_score,
            object_accuracy=object_score,
            composition_spatial=composition_score,
            color_lighting=color_score,
            fine_details=detail_score,
            total_score=total,
        )
