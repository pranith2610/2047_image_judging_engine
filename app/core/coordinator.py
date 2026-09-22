"""
Judging Coordinator for THE 2047 engine.
Assembles analysis, comparison, scoring, evidence generation, ranking, and session management into a production pipeline.
Optimized for 52-60 participants with batch chunking, resume support, single retry, and session persistence.
"""
import time
import json
import uuid
import hashlib
import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict

from app.config import SESSIONS_DIR, CATEGORY_WEIGHTS
from app.models.schemas import (
    ReferenceProfile,
    ParticipantProfile,
    EvaluationResult,
    RankedParticipant,
    BatchEvaluationResponse,
    StructuredEvidence,
    AuditMetadata,
    JudgingSession,
    ParticipantStatusInfo,
)
from app.core.analyzers.base import BaseReferenceAnalyzer, BaseParticipantAnalyzer
from app.core.analyzers.reference import BaselineReferenceAnalyzer
from app.core.analyzers.participant import BaselineParticipantAnalyzer
from app.core.comparators.base import BaseFeatureComparator
from app.core.comparators.comparator import BaselineFeatureComparator
from app.core.scoring.base import BaseScoringEngine
from app.core.scoring.engine import WeightedScoringEngine
from app.core.ranking.base import BaseRankingEngine
from app.core.ranking.engine import StandardRankingEngine
from app.core.explanations.base import BaseExplanationEngine
from app.core.explanations.engine import RuleBasedExplanationEngine, validate_explanation_consistency
from app.core.calibration.analyzer import CalibrationAnalyzer
from app.models.schemas import (
    HumanEvaluation,
    HumanScoreOverride,
    CalibrationStats,
    ABConfigSimulationRequest,
    ABConfigSimulationResponse,
)


class JudgingCoordinator:
    """
    Coordinates the multi-stage judging process.
    Reuses initialized models and analyzers for high performance across up to 60 participants.
    Supports resumable batch processing, single retries, duplicate detection, and session persistence.
    """

    def __init__(
        self,
        reference_analyzer: Optional[BaseReferenceAnalyzer] = None,
        participant_analyzer: Optional[BaseParticipantAnalyzer] = None,
        comparator: Optional[BaseFeatureComparator] = None,
        scoring_engine: Optional[BaseScoringEngine] = None,
        ranking_engine: Optional[BaseRankingEngine] = None,
        explanation_engine: Optional[BaseExplanationEngine] = None,
        calibration_analyzer: Optional[CalibrationAnalyzer] = None,
    ):
        self.ref_analyzer = reference_analyzer or BaselineReferenceAnalyzer()
        self.part_analyzer = participant_analyzer or BaselineParticipantAnalyzer()
        self.comparator = comparator or BaselineFeatureComparator()
        self.scoring_engine = scoring_engine or WeightedScoringEngine()
        self.ranking_engine = ranking_engine or StandardRankingEngine()
        self.explanation_engine = explanation_engine or RuleBasedExplanationEngine()
        self.calibration_analyzer = calibration_analyzer or CalibrationAnalyzer()

    @staticmethod
    def compute_image_hash(image_path: Path) -> Tuple[str, str]:
        """Calculate SHA-256 and MD5 hashes of the image file."""
        if not image_path.exists():
            return "", ""
        data = image_path.read_bytes()
        sha256 = hashlib.sha256(data).hexdigest()
        md5 = hashlib.md5(data).hexdigest()
        return sha256, md5

    def process_reference(self, image_path: Path, image_id: Optional[str] = None) -> ReferenceProfile:
        """Analyze reference image and generate structured ReferenceProfile."""
        return self.ref_analyzer.analyze(image_path, image_id)

    def evaluate_participant(
        self,
        image_path: Path,
        participant_id: str,
        reference_profile: ReferenceProfile,
    ) -> EvaluationResult:
        """Evaluate a single participant image against the reference profile."""
        # 1. Extract participant profile
        part_profile = self.part_analyzer.analyze(image_path, participant_id, reference_profile)

        # 2. Compare features across all 5 dimensions
        metrics = self.comparator.compare(reference_profile, part_profile)

        # 3. Calculate category scores (summing to exact max 100)
        scores = self.scoring_engine.calculate_scores(metrics, reference_profile)

        # 4. Generate structured evidence data for explanation layer and dossier UI
        evidence = StructuredEvidence(
            participant_id=participant_id,
            final_score=scores.total_score,
            semantic_score=scores.semantic_similarity,
            object_score=scores.object_accuracy,
            composition_score=scores.composition_spatial,
            color_lighting_score=scores.color_lighting,
            detail_score=scores.fine_details,
            matched_elements=metrics.matched_elements,
            partially_matched_elements=metrics.partially_matched_elements,
            missing_elements=metrics.missed_elements,
            unexpected_elements=metrics.extraneous_elements,
            composition_differences=metrics.composition_differences,
            color_differences=metrics.color_differences,
            detail_differences=metrics.detail_differences,
            strengths=metrics.strengths,
        )

        # 5. Generate comprehensive judging report via Explanation Engine
        judging_report = self.explanation_engine.generate_judging_report(
            reference_profile, part_profile, metrics, scores
        )

        # 6. Compute image hash
        sha256_hash, _ = self.compute_image_hash(image_path)

        return EvaluationResult(
            participant_id=participant_id,
            filename=image_path.name,
            scores=scores,
            explanation=judging_report.overall_explanation,
            category_reasons=judging_report.category_explanations,
            comparison_metrics=metrics,
            evidence=evidence,
            judging_report=judging_report,
            image_hash=sha256_hash,
        )

    def evaluate_single_participant(
        self,
        ref_image_path: Path,
        participant_id: str,
        participant_path: Path,
        reference_profile: Optional[ReferenceProfile] = None,
    ) -> EvaluationResult:
        """
        Evaluate or retry an individual participant recreation.
        Enables single-item retry without reprocessing the entire batch.
        """
        ref_prof = reference_profile or self.process_reference(ref_image_path)
        return self.evaluate_participant(participant_path, participant_id, ref_prof)

    def evaluate_batch(
        self,
        ref_image_path: Path,
        participants: List[Tuple[str, Path]],  # List of (participant_id, image_path)
        ref_id: Optional[str] = None,
        existing_evaluations: Optional[List[EvaluationResult]] = None,
    ) -> BatchEvaluationResponse:
        """
        Execute batch competition evaluation for up to 60 participants.
        Supports duplicate detection, resuming from existing evaluations, and audit verification.
        """
        start_time = time.time()

        # Step 1: Analyze reference image (computed once, reused for all participants)
        ref_profile = self.process_reference(ref_image_path, ref_id)
        ref_sha, _ = self.compute_image_hash(ref_image_path)

        # Step 2: Build evaluations map with existing ones for resume support
        eval_map: Dict[str, EvaluationResult] = {}
        if existing_evaluations:
            for ev in existing_evaluations:
                eval_map[ev.participant_id] = ev

        participant_hashes: Dict[str, str] = {}
        duplicate_flags: Dict[str, str] = {}
        seen_hashes: Dict[str, str] = {}  # sha -> first participant_id

        # Step 3: Evaluate remaining pending participants in memory-safe manner
        for p_id, p_path in participants:
            p_sha, _ = self.compute_image_hash(p_path)
            participant_hashes[p_id] = p_sha

            if p_sha in seen_hashes:
                duplicate_flags[p_id] = seen_hashes[p_sha]
            else:
                seen_hashes[p_sha] = p_id

            if p_id in eval_map:
                # Update duplicate status for existing evaluation if applicable
                ev = eval_map[p_id]
                if p_id in duplicate_flags:
                    ev.is_duplicate = True
                    ev.duplicate_of = duplicate_flags[p_id]
                ev.image_hash = p_sha
                continue

            try:
                eval_result = self.evaluate_participant(p_path, p_id, ref_profile)
                if p_id in duplicate_flags:
                    eval_result.is_duplicate = True
                    eval_result.duplicate_of = duplicate_flags[p_id]
                eval_map[p_id] = eval_result
            except Exception as exc:
                # Fault tolerance: Do not crash entire batch on a single corrupted image
                # Create fallback zeroed evaluation with failure explanation
                from app.models.schemas import CategoryScores
                zero_scores = CategoryScores(
                    semantic_similarity=0.0,
                    object_accuracy=0.0,
                    composition_spatial=0.0,
                    color_lighting=0.0,
                    fine_details=0.0,
                    total_score=0.0,
                )
                fallback_ev = EvaluationResult(
                    participant_id=p_id,
                    filename=p_path.name,
                    scores=zero_scores,
                    explanation=f"Analysis failed for {p_id}: {str(exc)}",
                    category_reasons={"error": f"Evaluation exception: {str(exc)}"},
                    image_hash=p_sha,
                )
                eval_map[p_id] = fallback_ev

        evaluations = list(eval_map.values())

        # Step 4: Validate score-explanation consistency across all generated reports
        consistency_issues = 0
        for ev in evaluations:
            if ev.judging_report:
                warnings = validate_explanation_consistency(ev.scores, ev.judging_report)
                if warnings:
                    consistency_issues += len(warnings)

        # Step 5: Rank participants with proper tie handling and duplicate forwarding
        ranked = self.ranking_engine.rank(evaluations)

        duration = round(time.time() - start_time, 2)

        # Count overrides and flags
        overrides_count = sum(1 for ev in evaluations if ev.human_override is not None)
        flagged_count = sum(1 for ev in evaluations if ev.flagged_for_review)

        # Step 6: Generate Comprehensive Audit Metadata for event organizers
        audit = AuditMetadata(
            engine_version="THE-2047-PROD-v1.0",
            timestamp=datetime.datetime.utcnow().isoformat() + "Z",
            total_evaluated=len(ranked),
            category_weights=CATEGORY_WEIGHTS,
            execution_duration_sec=duration,
            scoring_hash=uuid.uuid4().hex[:16],
            reference_hash=ref_sha,
            participant_hashes=participant_hashes,
            duplicate_flags=duplicate_flags,
            analyzer_versions={
                "reference_analyzer": "BaselineReferenceAnalyzer-v1.0",
                "participant_analyzer": "BaselineParticipantAnalyzer-v1.0",
                "comparator": "BaselineFeatureComparator-v1.0",
                "scoring_engine": "WeightedScoringEngine-v1.0",
                "ranking_engine": "StandardRankingEngine-v1.0",
                "explanation_engine": "RuleBasedExplanationEngine-v1.0",
                "calibration_analyzer": "CalibrationAnalyzer-v1.0",
            },
            consistency_checks_passed=(consistency_issues == 0),
            consistency_issues_count=consistency_issues,
            human_overrides_count=overrides_count,
            flagged_for_review_count=flagged_count,
        )

        return BatchEvaluationResponse(
            reference_id=ref_profile.image_id,
            reference_profile=ref_profile,
            total_participants=len(ranked),
            ranked_results=ranked,
            processing_time_seconds=duration,
            audit_metadata=audit,
        )

    def run_calibration_analysis(
        self,
        human_evals: List[HumanEvaluation],
        eval_results: List[EvaluationResult],
    ) -> CalibrationStats:
        """Run statistical calibration analysis comparing human evaluator scores vs AI scores."""
        return self.calibration_analyzer.analyze(human_evals, eval_results)

    def simulate_ab_weights(
        self,
        request: ABConfigSimulationRequest,
        eval_results: List[EvaluationResult],
    ) -> ABConfigSimulationResponse:
        """Simulate alternative category scoring weights against human ratings."""
        return self.calibration_analyzer.simulate_ab_weights(request, eval_results)

    # ==========================================
    # SESSION PERSISTENCE & RESUME HELPERS
    # ==========================================
    def save_session(self, session: JudgingSession, filepath: Optional[Path] = None) -> Path:
        """Save a judging session to disk as formatted JSON."""
        target_path = filepath or (SESSIONS_DIR / f"{session.session_id}.json")
        target_path.write_text(session.model_dump_json(indent=2), encoding="utf-8")
        return target_path

    def load_session(self, session_id_or_path: str) -> JudgingSession:
        """Load a judging session from disk by ID or direct file path."""
        p = Path(session_id_or_path)
        if not p.exists() or not p.is_file():
            p = SESSIONS_DIR / f"{session_id_or_path}.json"
        if not p.exists():
            raise FileNotFoundError(f"Session '{session_id_or_path}' not found.")

        content = p.read_text(encoding="utf-8")
        data = json.loads(content)
        return JudgingSession.model_validate(data)

    def list_saved_sessions(self) -> List[dict]:
        """List all saved competition judging sessions."""
        sessions = []
        for file in SESSIONS_DIR.glob("*.json"):
            try:
                content = json.loads(file.read_text(encoding="utf-8"))
                sessions.append({
                    "session_id": content.get("session_id", file.stem),
                    "session_name": content.get("session_name", "Untitled Session"),
                    "created_at": content.get("created_at", ""),
                    "updated_at": content.get("updated_at", ""),
                    "status": content.get("status", "UNKNOWN"),
                    "total_participants": len(content.get("ranked_results", [])),
                    "file_path": str(file),
                })
            except Exception:
                continue
        return sorted(sessions, key=lambda s: s.get("updated_at", ""), reverse=True)
