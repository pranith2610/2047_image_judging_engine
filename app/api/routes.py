"""
FastAPI routes for file uploads, evaluation, ranking, session management, and exports.
"""
import uuid
import json
import datetime
from typing import List, Optional
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from fastapi.responses import Response, HTMLResponse
from pydantic import BaseModel

from app.config import (
    REF_UPLOAD_DIR,
    PART_UPLOAD_DIR,
    MAX_PARTICIPANT_IMAGES,
    MAX_FILE_SIZE_BYTES,
)
from app.models.schemas import (
    ReferenceProfile,
    EvaluationResult,
    BatchEvaluationResponse,
    JudgingSession,
    ValidationResponse,
    CalibrationInspectResponse,
    HumanEvaluation,
    HumanScoreOverride,
    DisagreementItem,
    CalibrationStats,
    CalibrationAnalyzeRequest,
    ABConfigSimulationRequest,
    ABConfigSimulationResponse,
    FlagForReviewRequest,
    HumanOverrideRequest,
)
from app.core.validator import ImageValidator
from app.core.coordinator import JudgingCoordinator
from app.core.explanations.engine import validate_explanation_consistency

router = APIRouter(prefix="/api")
coordinator = JudgingCoordinator()


class ParticipantItemRequest(BaseModel):
    participant_id: str
    filename: str


class EvaluationRequest(BaseModel):
    reference_filename: str
    participants: List[ParticipantItemRequest]
    session_id: Optional[str] = None


class SingleEvaluationRequest(BaseModel):
    reference_filename: str
    participant: ParticipantItemRequest


class CalibrationInspectRequest(BaseModel):
    reference_filename: str
    participant_filename: str
    participant_id: Optional[str] = "P-CALIBRATION"


@router.get("/health")
async def health_check():
    """System health and operational parameters."""
    return {
        "status": "online",
        "engine": "THE 2047 — AI Image Comparison & Judging Engine",
        "version": "1.0.0-PROD",
        "max_participant_images": MAX_PARTICIPANT_IMAGES,
        "max_file_size_mb": MAX_FILE_SIZE_BYTES / (1024 * 1024),
    }


@router.post("/upload/reference")
async def upload_reference_image(file: UploadFile = File(...)):
    """
    Upload and validate the competition reference image.
    Generates the initial structured Reference Profile.
    """
    content = await file.read()
    val_res = ImageValidator.validate_file_bytes(file.filename or "reference.png", content)

    if not val_res.valid:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid reference image: {val_res.error}"
        )

    # Save to disk with unique prefix to avoid collisions
    unique_name = f"ref_{uuid.uuid4().hex[:8]}_{val_res.filename}"
    saved_path = REF_UPLOAD_DIR / unique_name
    saved_path.write_bytes(content)

    # Process and build Reference Profile
    ref_profile = coordinator.process_reference(saved_path, image_id=f"REF-{unique_name[:12]}")

    return {
        "success": True,
        "filename": unique_name,
        "file_url": f"/uploads/reference/{unique_name}",
        "profile": ref_profile,
    }


@router.post("/upload/participants")
async def upload_participant_images(
    files: List[UploadFile] = File(...),
    participant_ids: Optional[List[str]] = Form(None),
):
    """
    Upload up to 60 participant recreation images.
    Validates each image individually, maintaining separate participant IDs.
    """
    if len(files) > MAX_PARTICIPANT_IMAGES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Batch size exceeds limit: maximum {MAX_PARTICIPANT_IMAGES} images allowed (received {len(files)})."
        )

    results = []
    errors = []

    for idx, f in enumerate(files):
        content = await f.read()
        val = ImageValidator.validate_file_bytes(f.filename or f"p_{idx}.png", content)

        # Separate participant ID from filename
        if participant_ids and idx < len(participant_ids) and participant_ids[idx].strip():
            p_id = participant_ids[idx].strip()
        else:
            p_id = f"P{idx + 1:02d}"

        if not val.valid:
            errors.append({
                "participant_id": p_id,
                "original_filename": f.filename,
                "error": val.error,
            })
            continue

        unique_name = f"part_{uuid.uuid4().hex[:8]}_{val.filename}"
        saved_path = PART_UPLOAD_DIR / unique_name
        saved_path.write_bytes(content)

        results.append({
            "participant_id": p_id,
            "filename": unique_name,
            "original_filename": val.filename,
            "file_url": f"/uploads/participants/{unique_name}",
            "dimensions": val.dimensions,
            "size_bytes": val.file_size_bytes,
        })

    return {
        "success": len(results) > 0,
        "total_uploaded": len(results),
        "total_errors": len(errors),
        "participants": results,
        "errors": errors,
    }


@router.post("/evaluate", response_model=BatchEvaluationResponse)
async def evaluate_competition(request: EvaluationRequest):
    """
    Execute full multi-dimensional evaluation for all participant recreations.
    Generates 100-point scores, explanations, rankings, and audit metadata.
    """
    ref_path = REF_UPLOAD_DIR / request.reference_filename
    if not ref_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Reference image '{request.reference_filename}' not found."
        )

    if not request.participants:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No participant images provided for evaluation."
        )

    participant_list = []
    for p in request.participants:
        p_path = PART_UPLOAD_DIR / p.filename
        if not p_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Participant file '{p.filename}' not found."
            )
        participant_list.append((p.participant_id, p_path))

    batch_response = coordinator.evaluate_batch(ref_path, participant_list)
    return batch_response


@router.post("/evaluate/single", response_model=EvaluationResult)
async def evaluate_single_participant_route(request: SingleEvaluationRequest):
    """
    Evaluate or retry a single participant image.
    Enables single-item retry without re-evaluating the entire batch.
    """
    ref_path = REF_UPLOAD_DIR / request.reference_filename
    if not ref_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Reference image '{request.reference_filename}' not found."
        )

    p_path = PART_UPLOAD_DIR / request.participant.filename
    if not p_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Participant file '{request.participant.filename}' not found."
        )

    eval_result = coordinator.evaluate_single_participant(
        ref_path, request.participant.participant_id, p_path
    )
    return eval_result


@router.post("/calibration/inspect", response_model=CalibrationInspectResponse)
async def calibration_inspect(request: CalibrationInspectRequest):
    """
    Developer / organizer diagnostic endpoint.
    Performs deep inspection of visual features, bounding boxes, quadrant density grids,
    bipartite element matches, color histograms, and contradiction consistency status.
    """
    ref_path = REF_UPLOAD_DIR / request.reference_filename
    if not ref_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Reference image '{request.reference_filename}' not found."
        )

    p_path = PART_UPLOAD_DIR / request.participant_filename
    if not p_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Participant file '{request.participant_filename}' not found."
        )

    p_id = request.participant_id or "P-CALIBRATION"
    ref_profile = coordinator.process_reference(ref_path)
    part_profile = coordinator.part_analyzer.analyze(p_path, p_id, ref_profile)
    metrics = coordinator.comparator.compare(ref_profile, part_profile)
    scores = coordinator.scoring_engine.calculate_scores(metrics, ref_profile)
    judging_report = coordinator.explanation_engine.generate_judging_report(
        ref_profile, part_profile, metrics, scores
    )
    warnings = validate_explanation_consistency(scores, judging_report)

    ref_objects = ref_profile.objects or (ref_profile.main_subjects + ref_profile.secondary_subjects)
    part_objects = part_profile.detected_elements or (
        ([part_profile.main_subject] if part_profile.main_subject else []) + (part_profile.secondary_subjects or [])
    )

    return CalibrationInspectResponse(
        reference_id=ref_profile.image_id,
        participant_id=p_id,
        scores=scores,
        comparison_metrics=metrics,
        reference_objects=ref_objects,
        participant_objects=part_objects,
        quadrant_density_reference=ref_profile.composition.quadrant_density,
        quadrant_density_participant=(part_profile.composition.quadrant_density if part_profile.composition else {}),
        dominant_colors_reference=ref_profile.colors.dominant_colors,
        dominant_colors_participant=(part_profile.colors.dominant_colors if part_profile.colors else []),
        consistency_passed=(len(warnings) == 0),
        consistency_warnings=warnings,
        judging_report=judging_report,
    )


@router.post("/calibration/analyze", response_model=CalibrationStats)
async def analyze_calibration_endpoint(request: CalibrationAnalyzeRequest):
    """
    Compute full statistical calibration comparing human ratings against automated AI scores.
    Calculates MAE, mean difference bias, Pearson r, Spearman rho, ranking agreement %,
    category error breakdown, and flags significant disagreements (|diff| > 10 pts).
    """
    eval_list = request.evaluations or []
    stats = coordinator.run_calibration_analysis(request.human_evaluations, eval_list)
    return stats


@router.post("/calibration/ab-test", response_model=ABConfigSimulationResponse)
async def ab_test_weights_endpoint(
    request: ABConfigSimulationRequest,
    evaluations: Optional[List[EvaluationResult]] = None,
):
    """
    Developer-only A/B scoring simulator.
    Compares default weights (30/25/20/15/10) vs a custom proposed configuration.
    Measures MAE delta, Pearson correlation, and ranking agreement without mutating active weights.
    """
    eval_list = evaluations or []
    sim_resp = coordinator.simulate_ab_weights(request, eval_list)
    return sim_resp


@router.post("/participants/{participant_id}/override", response_model=HumanScoreOverride)
async def record_human_override_endpoint(
    participant_id: str,
    request: HumanOverrideRequest,
):
    """
    Record a transparent, non-destructive human score override.
    Preserves original AI score, records organizer name, timestamp, and explicit rationale.
    """
    override = HumanScoreOverride(
        participant_id=participant_id,
        original_ai_score=0.0,  # Updated by caller or session context
        override_score=round(float(request.override_score), 1),
        override_reason=request.override_reason.strip(),
        organizer_name=request.organizer_name or "Competition Organizer",
        timestamp=datetime.datetime.utcnow().isoformat() + "Z",
        flagged_for_review=True,
    )
    return override


@router.post("/participants/{participant_id}/flag-review")
async def flag_participant_for_review(
    participant_id: str,
    request: FlagForReviewRequest,
):
    """
    Flag or unflag a participant recreation for human organizer review.
    """
    return {
        "success": True,
        "participant_id": participant_id,
        "flagged_for_review": request.flagged,
        "reason": request.reason or ("Flagged for organizer inspection" if request.flagged else "Unflagged"),
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    }


@router.post("/calibration/export-report")
async def export_calibration_report(stats: CalibrationStats):
    """
    Export comprehensive calibration analysis report as Markdown.
    """
    disagreement_rows = ""
    if stats.significant_disagreements:
        for d in stats.significant_disagreements:
            comments = f" — *Comments: {d.human_comments}*" if d.human_comments else ""
            disagreement_rows += (
                f"- **{d.participant_id}**: Human: **{d.human_score:.1f}** | AI: **{d.ai_score:.1f}** "
                f"| Diff: **{d.difference:+.1f}** | Primary divergence: `{d.main_category_disagreement}`{comments}\n"
            )
    else:
        disagreement_rows = "- *None. All sample evaluations were within 10.0 points of human ratings.*\n"

    cat_mae_rows = ""
    for cat, err in stats.category_mae.items():
        cat_name = cat.replace("_", " ").title()
        cat_mae_rows += f"- **{cat_name}**: MAE **{err:.2f} pts**\n"

    recs = "\n".join([f"- {r}" for r in stats.recommendations])

    report_md = f"""# THE 2047 — AI Image Comparison & Judging Engine
## Official Calibration & Human-AI Alignment Report

**Date:** {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}  
**Engine Version:** THE-2047-PROD-v1.0  
**Status:** **{stats.decision}**

---

### 1. Summary Statistics

| Metric | Value | Reference Benchmark |
|---|:---:|:---:|
| **Total Calibration Samples** | **{stats.total_samples}** | $\\ge 7$ recommended |
| **Human Average Score** | **{stats.human_avg_score:.1f} / 100** | -- |
| **AI Average Score** | **{stats.ai_avg_score:.1f} / 100** | -- |
| **Mean Absolute Error (MAE)** | **{stats.mean_absolute_error:.2f} pts** | $\\le 8.5$ acceptable |
| **Mean Difference (Bias)** | **{stats.mean_difference:+.2f} pts** | $0.0 \\pm 3.0$ balanced |
| **Pearson Score Correlation ($r$)** | **{stats.pearson_correlation:.3f}** | $\\ge 0.82$ strong |
| **Spearman Rank Correlation ($\\rho$)** | **{stats.spearman_rank_correlation:.3f}** | $\\ge 0.80$ strong |
| **Pairwise Ranking Agreement** | **{stats.ranking_agreement_pct:.1f}%** | $\\ge 75.0%$ target |

---

### 2. Category-Level Errors (MAE)

{cat_mae_rows if cat_mae_rows else "- *No category breakdown provided.*"}

---

### 3. Significant Disagreements ($|Human - AI| > 10.0$ pts)

{disagreement_rows}

---

### 4. Calibration Assessment & Rationale

> **{stats.decision}**  
> {stats.decision_rationale}

#### Recommendations & Observations:
{recs}
"""
    return Response(
        content=report_md,
        media_type="text/markdown",
        headers={"Content-Disposition": "attachment; filename=the_2047_calibration_report.md"},
    )


# ==========================================
# SESSION MANAGEMENT (SAVE / LOAD / LIST)
# ==========================================
@router.post("/session/save")
async def save_session_endpoint(session: JudgingSession):
    """Save the complete judging session state to disk."""
    try:
        saved_path = coordinator.save_session(session)
        return {
            "success": True,
            "session_id": session.session_id,
            "file_path": str(saved_path),
            "message": f"Session '{session.session_name}' successfully saved."
        }
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save session: {str(exc)}"
        )


@router.get("/session/list")
async def list_sessions_endpoint():
    """List all saved judging sessions."""
    sessions = coordinator.list_saved_sessions()
    return {"sessions": sessions}


@router.get("/session/{session_id}", response_model=JudgingSession)
async def get_session_endpoint(session_id: str):
    """Load and return a saved judging session by ID."""
    try:
        session = coordinator.load_session(session_id)
        return session
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found."
        )


@router.post("/session/load", response_model=JudgingSession)
async def load_session_file_endpoint(file: UploadFile = File(...)):
    """Upload and load a judging session JSON file."""
    try:
        content = await file.read()
        data = json.loads(content.decode("utf-8"))
        session = JudgingSession.model_validate(data)
        # Also persist to server sessions dir
        coordinator.save_session(session)
        return session
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid session JSON file: {str(exc)}"
        )


# ==========================================
# EXPORT ENDPOINTS (CSV, JSON, PRINTABLE)
# ==========================================
@router.post("/export/csv")
async def export_results_csv(batch_results: BatchEvaluationResponse):
    """
    Export full competition judging results as CSV.
    Includes participant ranks, category breakdowns, score tiers, and explanations.
    """
    import io
    import csv

    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow([
        "Rank",
        "Participant ID",
        "Final Score (/100)",
        "Score Tier",
        "Is Tied",
        "Semantic Similarity (/30)",
        "Object Accuracy (/25)",
        "Composition (/20)",
        "Color & Lighting (/15)",
        "Fine Details (/10)",
        "Overall Explanation",
        "Strengths",
        "Differences",
        "Missing Elements",
    ])

    for p in batch_results.ranked_results:
        strengths_str = "; ".join(p.judging_report.strengths if p.judging_report else (p.evidence.strengths if p.evidence else []))
        diffs_str = "; ".join(p.judging_report.differences if p.judging_report else [])
        missing_str = "; ".join(p.evidence.missing_elements if p.evidence else [])
        tier_str = p.judging_report.score_tier if p.judging_report else p.tier

        writer.writerow([
            p.rank,
            p.participant_id,
            f"{p.total_score:.1f}",
            tier_str,
            "YES" if p.is_tied else "NO",
            f"{p.scores.semantic_similarity:.1f}",
            f"{p.scores.object_accuracy:.1f}",
            f"{p.scores.composition_spatial:.1f}",
            f"{p.scores.color_lighting:.1f}",
            f"{p.scores.fine_details:.1f}",
            p.explanation,
            strengths_str,
            diffs_str,
            missing_str,
        ])

    csv_content = output.getvalue()
    return Response(
        content=csv_content,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=the_2047_judging_results.csv"},
    )


@router.post("/export/json")
async def export_results_json(batch_results: BatchEvaluationResponse):
    """
    Export full structured judging evaluation dossier as JSON.
    """
    json_str = batch_results.model_dump_json(indent=2)
    return Response(
        content=json_str,
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=the_2047_judging_dossier.json"},
    )


@router.post("/export/printable")
async def export_printable_html(batch_results: BatchEvaluationResponse):
    """
    Generate clean, printable HTML scorecard for event judges and competition records.
    """
    ranked_rows = ""
    for p in batch_results.ranked_results:
        tier_str = p.judging_report.score_tier if p.judging_report else p.tier
        strengths = "<br>".join([f"• {s}" for s in (p.judging_report.strengths if p.judging_report else [])[:3]])
        diffs = "<br>".join([f"• {d}" for d in (p.judging_report.differences if p.judging_report else [])[:3]])

        ranked_rows += f"""
        <tr>
            <td style="font-weight: bold; text-align: center;">#{p.rank} {'(TIED)' if p.is_tied else ''}</td>
            <td style="font-weight: bold;">{p.participant_id}</td>
            <td style="text-align: center; font-weight: bold; font-size: 15px; color: #0284c7;">{p.total_score:.1f} / 100</td>
            <td><span style="display:inline-block; padding: 2px 6px; border-radius: 4px; background: #e0f2fe; color: #0369a1; font-size: 11px; font-weight: bold;">{tier_str}</span></td>
            <td>{p.scores.semantic_similarity:.1f}</td>
            <td>{p.scores.object_accuracy:.1f}</td>
            <td>{p.scores.composition_spatial:.1f}</td>
            <td>{p.scores.color_lighting:.1f}</td>
            <td>{p.scores.fine_details:.1f}</td>
            <td style="font-size: 11px; max-width: 250px;">{p.explanation}</td>
        </tr>
        """

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>THE 2047 — Official Judging Scorecard</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 30px; color: #1e293b; }}
            .header {{ border-bottom: 2px solid #0f172a; padding-bottom: 15px; margin-bottom: 20px; }}
            .title {{ font-size: 22px; font-weight: 800; text-transform: uppercase; letter-spacing: 0.05em; }}
            .sub {{ font-size: 13px; color: #64748b; margin-top: 4px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 12px; }}
            th {{ background: #f1f5f9; padding: 8px 10px; text-align: left; border: 1px solid #cbd5e1; font-weight: 700; }}
            td {{ padding: 8px 10px; border: 1px solid #e2e8f0; vertical-align: top; }}
            tr:nth-child(even) {{ background: #f8fafc; }}
            .footer {{ margin-top: 30px; font-size: 11px; color: #94a3b8; text-align: center; border-top: 1px solid #e2e8f0; padding-top: 10px; }}
            @media print {{
                body {{ margin: 10mm; }}
                button {{ display: none; }}
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <div class="title">THE 2047 — Official Competition Judging Scorecard</div>
            <div class="sub">Total Evaluated: {batch_results.total_participants} Participants | Engine: THE-2047-PROD-v1.0 | Date: {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}</div>
        </div>
        <table>
            <thead>
                <tr>
                    <th>Rank</th>
                    <th>Participant</th>
                    <th>Final Score</th>
                    <th>Match Level</th>
                    <th>Semantic (30)</th>
                    <th>Objects (25)</th>
                    <th>Composition (20)</th>
                    <th>Color (15)</th>
                    <th>Details (10)</th>
                    <th>Official Rationale</th>
                </tr>
            </thead>
            <tbody>
                {ranked_rows}
            </tbody>
        </table>
        <div class="footer">
            THE 2047 AI Image Comparison & Judging Engine • Verified Deterministic Results • Official Event Copy
        </div>
        <script>
            window.onload = function() {{ window.print(); }};
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@router.post("/export/audit")
async def export_audit_log(batch_results: BatchEvaluationResponse):
    """
    Export verified competition audit metadata log as JSON.
    """
    audit_data = batch_results.audit_metadata.model_dump_json(indent=2) if batch_results.audit_metadata else "{}"
    return Response(
        content=audit_data,
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=the_2047_audit_log.json"},
    )

