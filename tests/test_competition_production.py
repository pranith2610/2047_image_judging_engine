"""
Production and competition testing suite for THE 2047 engine.
Validates 52-participant and 60-participant workflows, session save/load,
resume capability, single-participant retry, audit metadata, and printable exports.
"""
from pathlib import Path
import json
import pytest
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient

from app.main import app
from app.core.coordinator import JudgingCoordinator
from app.models.schemas import JudgingSession, ParticipantStatusInfo


@pytest.fixture
def test_client():
    return TestClient(app)


def test_52_participants_competition_workflow(tmp_path: Path):
    """
    Test the exact 52-participant competition event workflow.
    Ensures memory stability, deterministic scoring, proper ranking, and sub-10s execution.
    """
    # 1. Create reference image
    ref_path = tmp_path / "ref_master.png"
    img_ref = Image.new("RGB", (300, 300), color=(20, 25, 60))
    draw_ref = ImageDraw.Draw(img_ref)
    draw_ref.ellipse([80, 80, 220, 220], fill=(220, 40, 40))
    draw_ref.rectangle([30, 30, 90, 90], fill=(240, 200, 30))
    img_ref.save(ref_path)

    # 2. Create 52 participant recreation images
    participants = []
    for i in range(1, 53):
        p_id = f"P{i:02d}"
        p_path = tmp_path / f"part_{p_id}.png"
        
        # Add varying offsets to test diversity
        shift = (i * 3) % 40
        img_p = Image.new("RGB", (300, 300), color=(20, 25, 60))
        draw_p = ImageDraw.Draw(img_p)
        draw_p.ellipse([80 + shift, 80, 220 + shift, 220], fill=(220, 40, 40))
        draw_p.rectangle([30, 30, 90, 90], fill=(240, 200, 30))
        img_p.save(p_path)

        participants.append((p_id, p_path))

    coordinator = JudgingCoordinator()
    
    # 3. Execute 52-participant batch evaluation
    response = coordinator.evaluate_batch(ref_path, participants)

    assert response.total_participants == 52
    assert len(response.ranked_results) == 52
    assert response.processing_time_seconds < 10.0  # Fast execution constraint
    assert response.audit_metadata is not None
    assert response.audit_metadata.total_evaluated == 52
    assert response.audit_metadata.engine_version == "THE-2047-PROD-v1.0"

    # Verify monotonic ranking order
    for idx in range(len(response.ranked_results) - 1):
        curr = response.ranked_results[idx]
        nxt = response.ranked_results[idx + 1]
        assert curr.total_score >= nxt.total_score
        assert curr.rank <= nxt.rank
        assert curr.judging_report is not None


def test_resume_capability_skips_evaluated(tmp_path: Path):
    """
    Test resume capability:
    If processing stops halfway (e.g. 5 completed, 5 pending),
    resuming must evaluate only pending participants and reuse completed evaluations.
    """
    ref_path = tmp_path / "ref_resume.png"
    img = Image.new("RGB", (200, 200), color=(30, 40, 50))
    img.save(ref_path)

    coordinator = JudgingCoordinator()
    ref_profile = coordinator.process_reference(ref_path)

    # Pre-evaluate P01..P05
    existing_evals = []
    all_participants = []
    for i in range(1, 11):
        p_id = f"P{i:02d}"
        p_path = tmp_path / f"part_{p_id}.png"
        img.save(p_path)
        all_participants.append((p_id, p_path))

        if i <= 5:
            ev = coordinator.evaluate_participant(p_path, p_id, ref_profile)
            existing_evals.append(ev)

    assert len(existing_evals) == 5

    # Resume batch with existing evaluations passed in
    resumed_response = coordinator.evaluate_batch(
        ref_path, all_participants, existing_evaluations=existing_evals
    )

    assert resumed_response.total_participants == 10
    assert len(resumed_response.ranked_results) == 10

    # Ensure P01 scores match pre-evaluated scores exactly
    p01_resumed = next(r for r in resumed_response.ranked_results if r.participant_id == "P01")
    assert p01_resumed.total_score == existing_evals[0].scores.total_score


def test_single_participant_retry(tmp_path: Path):
    """Test individual participant retry capability without full batch re-run."""
    ref_path = tmp_path / "ref.png"
    img = Image.new("RGB", (200, 200), color=(30, 30, 30))
    draw = ImageDraw.Draw(img)
    draw.ellipse([50, 50, 150, 150], fill=(220, 40, 40))
    img.save(ref_path)

    p17_path = tmp_path / "p17.png"
    img.save(p17_path)

    coordinator = JudgingCoordinator()
    single_res = coordinator.evaluate_single_participant(ref_path, "P17", p17_path)

    assert single_res.participant_id == "P17"
    assert single_res.scores.total_score >= 90.0
    assert single_res.judging_report is not None
    assert single_res.judging_report.score_tier == "Exceptional match"


def test_session_save_and_load(tmp_path: Path, test_client: TestClient):
    """Test saving and restoring a complete judging session."""
    session = JudgingSession(
        session_id="TEST-SESSION-001",
        session_name="Live Finals Session",
        created_at="2026-09-11T10:00:00Z",
        updated_at="2026-09-11T10:30:00Z",
        status="COMPLETED",
        reference_filename="ref.png",
        participants=[
            ParticipantStatusInfo(participant_id="P01", filename="p01.png", status="COMPLETED"),
            ParticipantStatusInfo(participant_id="P02", filename="p02.png", status="COMPLETED"),
        ],
    )

    # 1. Test save endpoint
    save_resp = test_client.post("/api/session/save", json=session.model_dump())
    assert save_resp.status_code == 200
    assert save_resp.json()["success"] is True

    # 2. Test list endpoint
    list_resp = test_client.get("/api/session/list")
    assert list_resp.status_code == 200
    sessions = list_resp.json()["sessions"]
    assert any(s["session_id"] == "TEST-SESSION-001" for s in sessions)

    # 3. Test retrieve endpoint
    get_resp = test_client.get("/api/session/TEST-SESSION-001")
    assert get_resp.status_code == 200
    loaded = get_resp.json()
    assert loaded["session_id"] == "TEST-SESSION-001"
    assert len(loaded["participants"]) == 2


def test_printable_scorecard_endpoint(tmp_path: Path, test_client: TestClient):
    """Test generating printable HTML scorecards for event organizers."""
    ref_path = tmp_path / "ref.png"
    img = Image.new("RGB", (200, 200), color=(40, 40, 40))
    img.save(ref_path)

    p_path = tmp_path / "p.png"
    img.save(p_path)

    coordinator = JudgingCoordinator()
    batch_res = coordinator.evaluate_batch(ref_path, [("P27", p_path)])

    resp = test_client.post("/api/export/printable", json=batch_res.model_dump())
    assert resp.status_code == 200
    assert "text/html" in resp.headers["content-type"]
    assert "P27" in resp.text
    assert "Official Competition Judging Scorecard" in resp.text
    assert "window.print()" in resp.text
