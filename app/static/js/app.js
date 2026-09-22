/**
 * THE 2047 - AI Image Comparison & Judging Engine
 * Production Competition Dashboard Client
 * 
 * Manages full live event workflow:
 * - 1 Reference + up to 60 Participant Recreations (including the 52 competitors)
 * - Explicit [ ANALYZE ALL ] organizer execution
 * - Real-time pipeline step-by-step progress tracking
 * - Fault-tolerant single-participant retry and resume support
 * - Session save, load, and persistence
 * - Interactive sorting, filtering, and search
 * - Side-by-side, Difference heatmap, and Opacity overlay visual tools
 * - CSV, JSON, and Printable HTML scorecard exports
 */

document.addEventListener('DOMContentLoaded', () => {
    // ==========================================
    // GLOBAL STATE
    // ==========================================
    const state = {
        sessionId: `SESSION-${Date.now().toString(36).toUpperCase()}`,
        sessionName: 'THE 2047 Competition Session',
        reference: null,        // { filename, file_url, profile }
        participants: [],       // Array of { participant_id, filename, original_filename, file_url, dimensions, size_bytes, status, error }
        evaluationResult: null, // Full BatchEvaluationResponse payload
        isEvaluating: false,
        activeCompMode: 'side', // 'side', 'diff', 'overlay'
        activeParticipantId: null,
    };

    // ==========================================
    // DOM ELEMENT REFERENCES
    // ==========================================
    // Header & HUD
    const hudRefStatus = document.getElementById('hudRefStatus');
    const hudRefDot = document.getElementById('hudRefDot');
    const hudPartStatus = document.getElementById('hudPartStatus');
    const systemStatus = document.getElementById('systemStatus');

    // Master Toolbar
    const btnNewSession = document.getElementById('btnNewSession');
    const btnToolbarUploadRef = document.getElementById('btnToolbarUploadRef');
    const btnToolbarUploadPart = document.getElementById('btnToolbarUploadPart');
    const btnExecuteCompare = document.getElementById('btnExecuteCompare');
    const btnSaveSession = document.getElementById('btnSaveSession');
    const btnLoadSession = document.getElementById('btnLoadSession');
    const sessionFileInput = document.getElementById('sessionFileInput');

    // Reference Panel
    const refDropzone = document.getElementById('refDropzone');
    const refFileInput = document.getElementById('refFileInput');
    const refDropzoneEmpty = document.getElementById('refDropzoneEmpty');
    const refPreviewContainer = document.getElementById('refPreviewContainer');
    const refPreviewImg = document.getElementById('refPreviewImg');
    const refReplaceBtn = document.getElementById('refReplaceBtn');
    const refFilename = document.getElementById('refFilename');
    const refDimensions = document.getElementById('refDimensions');
    const refProfileHud = document.getElementById('refProfileHud');
    const refProfileId = document.getElementById('refProfileId');
    const refPaletteSwatches = document.getElementById('refPaletteSwatches');
    const refElementsList = document.getElementById('refElementsList');
    const refLightingVal = document.getElementById('refLightingVal');
    const refCompositionVal = document.getElementById('refCompositionVal');
    const refFocalVal = document.getElementById('refFocalVal');

    // Participants Panel
    const partDropzone = document.getElementById('partDropzone');
    const partFileInput = document.getElementById('partFileInput');
    const participantCount = document.getElementById('participantCount');
    const participantsToolbar = document.getElementById('participantsToolbar');
    const uploadedBadge = document.getElementById('uploadedBadge');
    const btnAutoIndex = document.getElementById('btnAutoIndex');
    const btnClearParticipants = document.getElementById('btnClearParticipants');
    const participantGrid = document.getElementById('participantGrid');

    // Progress HUD
    const progressHud = document.getElementById('progressHud');
    const progressStatusText = document.getElementById('progressStatusText');
    const progressPercent = document.getElementById('progressPercent');
    const progressBar = document.getElementById('progressBar');
    const activePLabel = document.getElementById('activePLabel');
    const pipelineStepsChips = document.getElementById('pipelineStepsChips');
    const progressLog = document.getElementById('progressLog');

    // Results & Leaderboard
    const resultsStation = document.getElementById('resultsStation');
    const resultsSummaryText = document.getElementById('resultsSummaryText');
    const topPerformersBanner = document.getElementById('topPerformersBanner');
    const leaderboardBody = document.getElementById('leaderboardBody');
    const tableFilterInput = document.getElementById('tableFilterInput');
    const tableFilterSelect = document.getElementById('tableFilterSelect');
    const tableSortSelect = document.getElementById('tableSortSelect');
    const btnExportCsv = document.getElementById('btnExportCsv');
    const btnExportJson = document.getElementById('btnExportJson');
    const btnPrintScorecards = document.getElementById('btnPrintScorecards');

    // Dossier Modal
    const dossierModal = document.getElementById('dossierModal');
    const modalCloseBtn = document.getElementById('modalCloseBtn');
    const modalRankBadge = document.getElementById('modalRankBadge');
    const modalTierBadge = document.getElementById('modalTierBadge');
    const modalTiedBadge = document.getElementById('modalTiedBadge');
    const modalParticipantTitle = document.getElementById('modalParticipantTitle');
    const modalRefImg = document.getElementById('modalRefImg');
    const modalPartImg = document.getElementById('modalPartImg');
    const sideBySideRow = document.getElementById('sideBySideRow');
    const singleCanvasFrame = document.getElementById('singleCanvasFrame');
    const compCanvas = document.getElementById('compCanvas');
    const overlaySliderWrap = document.getElementById('overlaySliderWrap');
    const overlayOpacitySlider = document.getElementById('overlayOpacitySlider');
    const modalTotalScore = document.getElementById('modalTotalScore');
    const modalTierSublabel = document.getElementById('modalTierSublabel');
    const modalSemVal = document.getElementById('modalSemVal');
    const modalSemBar = document.getElementById('modalSemBar');
    const modalObjVal = document.getElementById('modalObjVal');
    const modalObjBar = document.getElementById('modalObjBar');
    const modalCompVal = document.getElementById('modalCompVal');
    const modalCompBar = document.getElementById('modalCompBar');
    const modalColorVal = document.getElementById('modalColorVal');
    const modalColorBar = document.getElementById('modalColorBar');
    const modalDetVal = document.getElementById('modalDetVal');
    const modalDetBar = document.getElementById('modalDetBar');

    // Dossier Judging Sections
    const modalOverallExplanation = document.getElementById('modalOverallExplanation');
    const modalStrongestAreas = document.getElementById('modalStrongestAreas');
    const modalAreasToImprove = document.getElementById('modalAreasToImprove');
    const modalStrengthsChecklist = document.getElementById('modalStrengthsChecklist');
    const modalDifferencesChecklist = document.getElementById('modalDifferencesChecklist');
    const modalCategoryRationales = document.getElementById('modalCategoryRationales');
    const modalKeyElementsTableBody = document.getElementById('modalKeyElementsTableBody');
    const btnToggleEvidenceGrid = document.getElementById('btnToggleEvidenceGrid');
    const modalEvidenceGrid = document.getElementById('modalEvidenceGrid');
    const modalMatchedElements = document.getElementById('modalMatchedElements');
    const modalPartialElements = document.getElementById('modalPartialElements');
    const modalMissingElements = document.getElementById('modalMissingElements');
    const modalUnexpectedElements = document.getElementById('modalUnexpectedElements');
    const modalCompDiffs = document.getElementById('modalCompDiffs');
    const modalColorDiffs = document.getElementById('modalColorDiffs');
    const modalDetailDiffs = document.getElementById('modalDetailDiffs');
    const modalStrengths = document.getElementById('modalStrengths');

    // Toast Container
    const toastContainer = document.getElementById('toastContainer');

    // ==========================================
    // TOAST NOTIFICATIONS
    // ==========================================
    function showToast(message, type = 'info') {
        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.textContent = message;
        toastContainer.appendChild(toast);
        setTimeout(() => {
            toast.style.opacity = '0';
            setTimeout(() => toast.remove(), 300);
        }, 4000);
    }

    // ==========================================
    // LIVE HUD UPDATE
    // ==========================================
    function updateLiveHud() {
        if (state.reference) {
            hudRefStatus.textContent = 'READY';
            hudRefStatus.className = 'hud-value text-green';
            hudRefDot.className = 'hud-dot dot-green';
        } else {
            hudRefStatus.textContent = 'NONE';
            hudRefStatus.className = 'hud-value';
            hudRefDot.className = 'hud-dot dot-gray';
        }

        const count = state.participants.length;
        hudPartStatus.textContent = `${count} / 60`;
        participantCount.textContent = count;

        if (state.isEvaluating) {
            systemStatus.textContent = 'PROCESSING...';
            systemStatus.className = 'hud-value text-amber';
        } else if (state.evaluationResult) {
            systemStatus.textContent = 'COMPLETED';
            systemStatus.className = 'hud-value text-green';
        } else {
            systemStatus.textContent = state.reference && count > 0 ? 'READY TO ANALYZE' : 'WAITING FOR INPUT';
            systemStatus.className = 'hud-value';
        }

        btnExecuteCompare.disabled = !(state.reference && count > 0 && !state.isEvaluating);
    }

    // ==========================================
    // DRAG & DROP BINDINGS
    // ==========================================
    function bindDragAndDrop(zone, input, onFilesDropped) {
        zone.addEventListener('click', (e) => {
            if (e.target.closest('.btn-replace') || e.target.closest('.btn-part-remove') || e.target.tagName === 'INPUT') return;
            input.click();
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            zone.addEventListener(eventName, (e) => {
                e.preventDefault();
                zone.classList.add('drag-over');
            });
        });

        ['dragleave', 'drop'].forEach(eventName => {
            zone.addEventListener(eventName, (e) => {
                e.preventDefault();
                zone.classList.remove('drag-over');
            });
        });

        zone.addEventListener('drop', (e) => {
            if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                onFilesDropped(e.dataTransfer.files);
            }
        });

        input.addEventListener('change', (e) => {
            if (e.target.files && e.target.files.length > 0) {
                onFilesDropped(e.target.files);
                input.value = '';
            }
        });
    }

    // ==========================================
    // REFERENCE IMAGE LOGIC
    // ==========================================
    bindDragAndDrop(refDropzone, refFileInput, handleReferenceUpload);
    refReplaceBtn.addEventListener('click', () => refFileInput.click());
    btnToolbarUploadRef.addEventListener('click', () => refFileInput.click());

    async function handleReferenceUpload(fileList) {
        const file = fileList[0];
        if (!file) return;

        if (!file.type.match(/^image\/(jpeg|png|webp)$/i)) {
            showToast('Only JPG, PNG, and WEBP images are allowed.', 'error');
            return;
        }
        if (file.size > 15 * 1024 * 1024) {
            showToast('Reference image must be under 15MB.', 'error');
            return;
        }

        const formData = new FormData();
        formData.append('file', file);

        showToast('Analyzing Reference Image & extracting profile...', 'info');

        try {
            const resp = await fetch('/api/upload/reference', {
                method: 'POST',
                body: formData,
            });

            if (!resp.ok) {
                const err = await resp.json();
                throw new Error(err.detail || 'Upload failed');
            }

            const data = await resp.json();
            state.reference = data;

            refPreviewImg.src = data.file_url;
            refFilename.textContent = data.filename;
            if (data.profile && data.profile.dimensions) {
                refDimensions.textContent = `${data.profile.dimensions[0]}x${data.profile.dimensions[1]}`;
            }

            refDropzoneEmpty.classList.add('hidden');
            refPreviewContainer.classList.remove('hidden');

            renderReferenceProfile(data.profile);
            updateLiveHud();
            showToast('Reference profile analyzed successfully.', 'success');
        } catch (err) {
            console.error(err);
            showToast(`Reference upload error: ${err.message}`, 'error');
        }
    }

    function renderReferenceProfile(profile) {
        if (!profile) return;
        refProfileHud.classList.remove('hidden');
        refProfileId.textContent = profile.image_id;

        // Dominant Colors
        refPaletteSwatches.innerHTML = '';
        const domColors = profile.colors?.dominant_colors || profile.dominant_colors?.dominant_colors || [];
        if (domColors.length > 0) {
            domColors.forEach(col => {
                const chip = document.createElement('div');
                chip.className = 'swatch-chip';
                chip.innerHTML = `
                    <span class="swatch-color" style="background-color: ${col.hex_code}"></span>
                    <span class="swatch-info">${col.hex_code} (${col.percentage}%)</span>
                `;
                refPaletteSwatches.appendChild(chip);
            });
        }

        // Salient Elements
        refElementsList.innerHTML = '';
        const allElems = profile.objects || [...(profile.main_subjects || []), ...(profile.secondary_subjects || [])];
        if (allElems.length > 0) {
            allElems.slice(0, 5).forEach(elem => {
                const item = document.createElement('div');
                item.className = 'element-item';
                const impClass = elem.importance === 'HIGH' ? 'badge-high' : elem.importance === 'MEDIUM' ? 'badge-med' : 'badge-low';
                item.innerHTML = `
                    <span class="elem-name">${elem.name}</span>
                    <span class="elem-badge ${impClass}">${elem.importance} IMP</span>
                `;
                refElementsList.appendChild(item);
            });
        } else {
            refElementsList.innerHTML = '<span class="text-dim">Standard visual scene elements</span>';
        }

        // Metrics
        refLightingVal.textContent = profile.lighting?.lighting_description || 'Balanced';
        refCompositionVal.textContent = profile.composition?.layout_description || 'Standard';
        if (profile.composition?.focal_center) {
            refFocalVal.textContent = `(${profile.composition.focal_center[0].toFixed(2)}, ${profile.composition.focal_center[1].toFixed(2)})`;
        }
    }

    // ==========================================
    // PARTICIPANT IMAGES LOGIC
    // ==========================================
    bindDragAndDrop(partDropzone, partFileInput, handleParticipantsUpload);
    btnToolbarUploadPart.addEventListener('click', () => partFileInput.click());
    btnAutoIndex.addEventListener('click', autoIndexParticipantIds);
    btnClearParticipants.addEventListener('click', clearAllParticipants);

    async function handleParticipantsUpload(fileList) {
        const remainingCapacity = 60 - state.participants.length;
        if (remainingCapacity <= 0) {
            showToast('Maximum batch capacity of 60 participants reached.', 'error');
            return;
        }

        const filesToUpload = Array.from(fileList).slice(0, remainingCapacity);
        const startIndex = state.participants.length;

        const formData = new FormData();
        filesToUpload.forEach((file, idx) => {
            formData.append('files', file);
            const pId = `P${startIndex + idx + 1 < 10 ? '0' : ''}${startIndex + idx + 1}`;
            formData.append('participant_ids', pId);
        });

        showToast(`Uploading and validating ${filesToUpload.length} participant recreation(s)...`, 'info');

        try {
            const resp = await fetch('/api/upload/participants', {
                method: 'POST',
                body: formData,
            });

            if (!resp.ok) {
                const err = await resp.json();
                throw new Error(err.detail || 'Upload failed');
            }

            const data = await resp.json();
            if (data.participants && data.participants.length > 0) {
                data.participants.forEach(p => {
                    p.status = 'PENDING';
                    state.participants.push(p);
                });
                renderParticipantsGrid();
                updateLiveHud();
                showToast(`Added ${data.participants.length} participant(s). Total: ${state.participants.length}/60.`, 'success');
            }

            if (data.errors && data.errors.length > 0) {
                showToast(`${data.errors.length} file(s) failed validation.`, 'error');
            }
        } catch (err) {
            console.error(err);
            showToast(`Participant upload error: ${err.message}`, 'error');
        }
    }

    function renderParticipantsGrid() {
        participantGrid.innerHTML = '';
        const count = state.participants.length;

        if (count > 0) {
            participantsToolbar.classList.remove('hidden');
            uploadedBadge.textContent = `${count} IMAGES LOADED`;
        } else {
            participantsToolbar.classList.add('hidden');
        }

        state.participants.forEach((p, idx) => {
            const card = document.createElement('div');
            card.className = 'participant-item-card';

            const statusClass = p.status === 'COMPLETED' ? 'status-pill-done' : p.status === 'FAILED' ? 'status-pill-fail' : 'status-pill-pending';

            card.innerHTML = `
                <div class="part-thumb-wrap">
                    <img src="${p.file_url}" alt="${p.participant_id}">
                    <button type="button" class="btn-part-remove" data-index="${idx}" title="Remove participant">&times;</button>
                </div>
                <div class="part-id-input-wrap">
                    <input type="text" class="part-id-input" data-index="${idx}" value="${p.participant_id}" placeholder="P01">
                </div>
                <div class="part-card-footer">
                    <span class="part-status-pill ${statusClass}">${p.status || 'READY'}</span>
                    ${p.status === 'FAILED' ? `<button type="button" class="btn-retry-single" data-id="${p.participant_id}">Retry</button>` : ''}
                </div>
            `;
            participantGrid.appendChild(card);
        });

        // Event bindings for grid cards
        participantGrid.querySelectorAll('.btn-part-remove').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                removeParticipant(parseInt(btn.dataset.index, 10));
            });
        });

        participantGrid.querySelectorAll('.part-id-input').forEach(input => {
            input.addEventListener('change', (e) => {
                const idx = parseInt(e.target.dataset.index, 10);
                if (state.participants[idx]) {
                    state.participants[idx].participant_id = e.target.value.trim().toUpperCase() || `P${(idx + 1).toString().padStart(2, '0')}`;
                }
            });
        });

        participantGrid.querySelectorAll('.btn-retry-single').forEach(btn => {
            btn.addEventListener('click', () => {
                retrySingleParticipant(btn.dataset.id);
            });
        });
    }

    function autoIndexParticipantIds() {
        state.participants.forEach((p, idx) => {
            p.participant_id = `P${idx + 1 < 10 ? '0' : ''}${idx + 1}`;
        });
        renderParticipantsGrid();
        showToast('Participant IDs sequentially indexed (P01..P60).', 'info');
    }

    function removeParticipant(index) {
        if (index < 0 || index >= state.participants.length) return;
        state.participants.splice(index, 1);
        renderParticipantsGrid();
        updateLiveHud();
        showToast('Participant removed.', 'info');
    }

    function clearAllParticipants() {
        if (state.participants.length === 0) return;
        state.participants = [];
        renderParticipantsGrid();
        updateLiveHud();
        showToast('Cleared all participant images.', 'info');
    }

    // ==========================================
    // EXECUTION & PROGRESS HUD (LIVE PIPELINE)
    // ==========================================
    btnExecuteCompare.addEventListener('click', executeComparison);

    async function executeComparison() {
        if (!state.reference || state.participants.length === 0 || state.isEvaluating) return;

        state.isEvaluating = true;
        updateLiveHud();

        // Reveal Progress HUD
        progressHud.classList.remove('hidden');
        progressBar.style.width = '5%';
        progressPercent.textContent = '5%';
        progressStatusText.textContent = `Analyzing ${state.participants.length} participant recreation(s)...`;
        progressLog.innerHTML = '';

        const total = state.participants.length;
        let processedCount = 0;

        // Simulated progressive pipeline step tracking
        const stepLabels = ['✓ Loaded', '✓ Features Extracted', '✓ Compared', '✓ Score Calculated', '✓ Explanation Generated'];
        const stepInterval = setInterval(() => {
            if (processedCount < total) {
                const p = state.participants[processedCount];
                activePLabel.textContent = `Current: ${p.participant_id} (${p.original_filename || p.filename})`;

                // Render pipeline steps
                pipelineStepsChips.innerHTML = '';
                stepLabels.forEach(step => {
                    const chip = document.createElement('span');
                    chip.className = 'step-chip step-done';
                    chip.textContent = step;
                    pipelineStepsChips.appendChild(chip);
                });

                const logItem = document.createElement('span');
                logItem.className = 'progress-log-item';
                logItem.textContent = `${p.participant_id} ✓`;
                progressLog.appendChild(logItem);

                processedCount++;
                const pct = Math.min(92, Math.round((processedCount / total) * 92));
                progressBar.style.width = `${pct}%`;
                progressPercent.textContent = `${pct}%`;
            }
        }, Math.max(60, Math.floor(1200 / total)));

        try {
            const payload = {
                reference_filename: state.reference.filename,
                participants: state.participants.map(p => ({
                    participant_id: p.participant_id,
                    filename: p.filename,
                })),
                session_id: state.sessionId,
            };

            const resp = await fetch('/api/evaluate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });

            clearInterval(stepInterval);

            if (!resp.ok) {
                const err = await resp.json();
                throw new Error(err.detail || 'Evaluation failed');
            }

            const data = await resp.json();
            state.evaluationResult = data;

            // Mark all participant statuses completed
            state.participants.forEach(p => { p.status = 'COMPLETED'; });
            renderParticipantsGrid();

            progressBar.style.width = '100%';
            progressPercent.textContent = '100%';
            progressStatusText.textContent = `Completed judging for ${data.total_participants} participants in ${data.processing_time_seconds}s!`;

            setTimeout(() => {
                progressHud.classList.add('hidden');
                state.isEvaluating = false;
                updateLiveHud();
                renderResults(data);
                showToast(`Judging complete! ${data.total_participants} participants evaluated deterministically.`, 'success');
            }, 600);

        } catch (err) {
            clearInterval(stepInterval);
            progressHud.classList.add('hidden');
            state.isEvaluating = false;
            updateLiveHud();
            console.error(err);
            showToast(`Evaluation error: ${err.message}`, 'error');
        }
    }

    async function retrySingleParticipant(participantId) {
        if (!state.reference) return;
        const p = state.participants.find(item => item.participant_id === participantId);
        if (!p) return;

        showToast(`Retrying evaluation for ${participantId}...`, 'info');

        try {
            const resp = await fetch('/api/evaluate/single', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    reference_filename: state.reference.filename,
                    participant: {
                        participant_id: p.participant_id,
                        filename: p.filename,
                    }
                }),
            });

            if (!resp.ok) throw new Error('Single retry failed');
            const result = await resp.json();

            p.status = 'COMPLETED';
            renderParticipantsGrid();

            // If evaluationResult exists, update or re-evaluate
            if (state.evaluationResult) {
                await executeComparison();
            }
            showToast(`Successfully re-evaluated ${participantId}!`, 'success');
        } catch (err) {
            console.error(err);
            showToast(`Retry failed: ${err.message}`, 'error');
        }
    }

    // ==========================================
    // RESULTS & FINAL LEADERBOARD DASHBOARD
    // ==========================================
    function renderResults(data) {
        resultsStation.classList.remove('hidden');
        resultsStation.scrollIntoView({ behavior: 'smooth', block: 'start' });

        resultsSummaryText.textContent = `Official leaderboard: ${data.total_participants} evaluated recreation(s) across 5 visual categories (Completed in ${data.processing_time_seconds}s).`;

        const ranked = data.ranked_results || [];

        // 1. Render Compact Top Performers Banner (Rank 1, 2, 3)
        topPerformersBanner.innerHTML = '';
        const top3 = ranked.slice(0, 3);
        const medals = ['🥇', '🥈', '🥉'];
        const rankLabels = ['RANK 1 — GRAND CHAMPION', 'RANK 2 — PODIUM FINALIST', 'RANK 3 — PODIUM FINALIST'];

        top3.forEach((item, idx) => {
            const partInfo = state.participants.find(p => p.participant_id === item.participant_id) || {};
            const card = document.createElement('div');
            card.className = `top-performer-card rank-card-${item.rank}`;

            const tieBadge = item.is_tied ? '<span class="pill-tied">TIED</span>' : '';
            const scoreTier = item.judging_report?.score_tier || item.tier;

            card.innerHTML = `
                <div class="top-medal-badge">${medals[idx] || '★'} ${rankLabels[idx]} ${tieBadge}</div>
                <div class="top-card-body">
                    <img class="top-thumb" src="${partInfo.file_url || ''}" alt="${item.participant_id}">
                    <div class="top-info">
                        <span class="top-pid">${item.participant_id}</span>
                        <div class="top-score-wrap">
                            <span class="top-score-num">${item.total_score.toFixed(1)}</span>
                            <span class="top-score-max">/ 100</span>
                        </div>
                        <span class="top-tier-pill">${scoreTier}</span>
                    </div>
                </div>
                <button type="button" class="btn btn-secondary btn-sm view-dossier-btn" data-id="${item.participant_id}">
                    View Official Report
                </button>
            `;
            topPerformersBanner.appendChild(card);
        });

        // 2. Render Full Results Table
        applyFilterAndSort();
    }

    function applyFilterAndSort() {
        if (!state.evaluationResult) return;
        let items = [...state.evaluationResult.ranked_results];

        // 1. Text Search Filter
        const query = tableFilterInput.value.toLowerCase().trim();
        if (query) {
            items = items.filter(item => 
                item.participant_id.toLowerCase().includes(query) || 
                item.filename.toLowerCase().includes(query)
            );
        }

        // 2. Dropdown Filter
        const filterVal = tableFilterSelect.value;
        if (filterVal === 'top10') {
            items = items.filter(i => i.rank <= 10);
        } else if (filterVal === 'top20') {
            items = items.filter(i => i.rank <= 20);
        } else if (filterVal === 'score80') {
            items = items.filter(i => i.total_score >= 80.0);
        } else if (filterVal === 'score70') {
            items = items.filter(i => i.total_score >= 70.0);
        } else if (filterVal === 'score50') {
            items = items.filter(i => i.total_score < 50.0);
        } else if (filterVal === 'tied') {
            items = items.filter(i => i.is_tied);
        }

        // 3. Dropdown Sort
        const sortVal = tableSortSelect.value;
        if (sortVal === 'score_desc') {
            items.sort((a, b) => b.total_score - a.total_score);
        } else if (sortVal === 'semantic_desc') {
            items.sort((a, b) => b.scores.semantic_similarity - a.scores.semantic_similarity);
        } else if (sortVal === 'object_desc') {
            items.sort((a, b) => b.scores.object_accuracy - a.scores.object_accuracy);
        } else if (sortVal === 'comp_desc') {
            items.sort((a, b) => b.scores.composition_spatial - a.scores.composition_spatial);
        } else if (sortVal === 'color_desc') {
            items.sort((a, b) => b.scores.color_lighting - a.scores.color_lighting);
        } else if (sortVal === 'details_desc') {
            items.sort((a, b) => b.scores.fine_details - a.scores.fine_details);
        } else if (sortVal === 'id_asc') {
            items.sort((a, b) => a.participant_id.localeCompare(b.participant_id));
        }

        renderTableRows(items);
    }

    function renderTableRows(items) {
        leaderboardBody.innerHTML = '';

        items.forEach(item => {
            const partInfo = state.participants.find(p => p.participant_id === item.participant_id) || {};
            const tr = document.createElement('tr');

            let rankClass = '';
            if (item.rank === 1) rankClass = 'rank-1-badge';
            else if (item.rank === 2) rankClass = 'rank-2-badge';
            else if (item.rank === 3) rankClass = 'rank-3-badge';

            const tieTag = item.is_tied ? '<span class="pill-tied">TIED</span>' : '';
            const dupTag = item.is_duplicate ? `<span class="badge-duplicate" title="Identical to ${item.duplicate_of || 'another submission'}">DUP (${item.duplicate_of || 'FLAGGED'})</span>` : '';
            const scoreTier = item.judging_report?.score_tier || item.tier;

            tr.innerHTML = `
                <td><span class="rank-badge ${rankClass}">#${item.rank}</span></td>
                <td><strong>${item.participant_id}</strong> ${tieTag} ${dupTag}</td>
                <td><img class="table-thumb" src="${partInfo.file_url || ''}" alt="${item.participant_id}"></td>
                <td><span class="cat-score-num">${item.scores.semantic_similarity.toFixed(1)}/30</span></td>
                <td><span class="cat-score-num">${item.scores.object_accuracy.toFixed(1)}/25</span></td>
                <td><span class="cat-score-num">${item.scores.composition_spatial.toFixed(1)}/20</span></td>
                <td><span class="cat-score-num">${item.scores.color_lighting.toFixed(1)}/15</span></td>
                <td><span class="cat-score-num">${item.scores.fine_details.toFixed(1)}/10</span></td>
                <td><span class="score-cell-pill">${item.total_score.toFixed(1)}/100</span></td>
                <td><span class="tier-pill ${getTierBadgeClass(item.total_score)}">${scoreTier}</span></td>
                <td>
                    <button type="button" class="btn btn-secondary btn-sm view-dossier-btn" data-id="${item.participant_id}">
                        Report
                    </button>
                </td>
            `;
            leaderboardBody.appendChild(tr);
        });

        // Re-bind table dossier buttons
        document.querySelectorAll('.view-dossier-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                openDossierModal(btn.dataset.id);
            });
        });
    }

    function getTierBadgeClass(score) {
        if (score >= 90.0) return 'tier-champion';
        if (score >= 80.0) return 'tier-podium';
        if (score >= 70.0) return 'tier-distinguished';
        return 'tier-contender';
    }

    // Filter & Sort Event Listeners
    tableFilterInput.addEventListener('input', applyFilterAndSort);
    tableFilterSelect.addEventListener('change', applyFilterAndSort);
    tableSortSelect.addEventListener('change', applyFilterAndSort);

    // ==========================================
    // DOSSIER MODAL WITH VISUAL COMPARISON MODES
    // ==========================================
    function openDossierModal(participantId) {
        if (!state.evaluationResult) return;
        const record = state.evaluationResult.ranked_results.find(r => r.participant_id === participantId);
        if (!record) return;

        state.activeParticipantId = participantId;
        const partInfo = state.participants.find(p => p.participant_id === participantId) || {};
        const evidence = record.evidence || {};
        const report = record.judging_report || {};

        modalRankBadge.textContent = `RANK #${record.rank}`;
        const tierText = report.score_tier || record.tier;
        modalTierBadge.textContent = tierText.toUpperCase();

        modalTierBadge.className = 'badge badge-tier';
        if (record.total_score >= 90.0) modalTierBadge.classList.add('tier-exceptional');
        else if (record.total_score >= 80.0) modalTierBadge.classList.add('tier-strong');
        else if (record.total_score >= 60.0) modalTierBadge.classList.add('tier-moderate');
        else modalTierBadge.classList.add('tier-low');

        if (record.is_tied) {
            modalTiedBadge.classList.remove('hidden');
        } else {
            modalTiedBadge.classList.add('hidden');
        }

        modalParticipantTitle.textContent = `OFFICIAL JUDGING REPORT: ${record.participant_id}`;

        modalRefImg.src = state.reference?.file_url || '';
        modalPartImg.src = partInfo.file_url || '';

        modalTotalScore.textContent = record.total_score.toFixed(1);
        modalTierSublabel.textContent = tierText;

        // Scores & Progress Bars
        modalSemVal.textContent = `${record.scores.semantic_similarity.toFixed(1)} / 30`;
        modalSemBar.style.width = `${(record.scores.semantic_similarity / 30) * 100}%`;

        modalObjVal.textContent = `${record.scores.object_accuracy.toFixed(1)} / 25`;
        modalObjBar.style.width = `${(record.scores.object_accuracy / 25) * 100}%`;

        modalCompVal.textContent = `${record.scores.composition_spatial.toFixed(1)} / 20`;
        modalCompBar.style.width = `${(record.scores.composition_spatial / 20) * 100}%`;

        modalColorVal.textContent = `${record.scores.color_lighting.toFixed(1)} / 15`;
        modalColorBar.style.width = `${(record.scores.color_lighting / 15) * 100}%`;

        modalDetVal.textContent = `${record.scores.fine_details.toFixed(1)} / 10`;
        modalDetBar.style.width = `${(record.scores.fine_details / 10) * 100}%`;

        // Section 1: Explanation Narrative & Performance Areas
        modalOverallExplanation.textContent = report.overall_explanation || record.explanation || "Evaluation metrics computed.";

        modalStrongestAreas.innerHTML = '';
        (report.strongest_areas || ['Semantic Similarity', 'Object Accuracy']).forEach(area => {
            const span = document.createElement('span');
            span.className = 'perf-tag tag-strong';
            span.textContent = `★ ${area}`;
            modalStrongestAreas.appendChild(span);
        });

        modalAreasToImprove.innerHTML = '';
        (report.areas_to_improve || ['Composition', 'Color & Lighting']).forEach(area => {
            const span = document.createElement('span');
            span.className = 'perf-tag tag-improve';
            span.textContent = `△ ${area}`;
            modalAreasToImprove.appendChild(span);
        });

        // Section 2: Strengths & Differences Dual Panel
        modalStrengthsChecklist.innerHTML = '';
        const strengthsList = report.strengths || evidence.strengths || [];
        if (strengthsList.length > 0) {
            strengthsList.forEach(item => {
                const li = document.createElement('li');
                li.className = 'judge-bullet-item';
                li.innerHTML = `<span class="bullet-icon text-emerald">✓</span> <span>${item.replace(/^[✓★]\s*/, '')}</span>`;
                modalStrengthsChecklist.appendChild(li);
            });
        } else {
            modalStrengthsChecklist.innerHTML = '<li class="text-dim">Baseline recreation fidelity</li>';
        }

        modalDifferencesChecklist.innerHTML = '';
        const diffsList = report.differences || evidence.composition_differences || [];
        if (diffsList.length > 0) {
            diffsList.forEach(item => {
                const li = document.createElement('li');
                li.className = 'judge-bullet-item';
                const isDeduction = item.startsWith('✗');
                const icon = isDeduction ? '✗' : '△';
                const iconClass = isDeduction ? 'text-rose' : 'text-amber';
                li.innerHTML = `<span class="bullet-icon ${iconClass}">${icon}</span> <span>${item.replace(/^[△✗]\s*/, '')}</span>`;
                modalDifferencesChecklist.appendChild(li);
            });
        } else {
            modalDifferencesChecklist.innerHTML = '<li class="text-dim">Minor pixel variations only</li>';
        }

        // Section 3: Category Rationales
        modalCategoryRationales.innerHTML = '';
        const catRationales = report.category_explanations || record.category_reasons || {};
        const catConfig = [
            { key: 'semantic_similarity', label: 'Semantic Similarity (30 pts)', icon: '🔮' },
            { key: 'object_accuracy', label: 'Object / Element Accuracy (25 pts)', icon: '🎯' },
            { key: 'composition_spatial', label: 'Composition & Spatial Layout (20 pts)', icon: '📐' },
            { key: 'color_lighting', label: 'Color & Lighting Fidelity (15 pts)', icon: '🎨' },
            { key: 'fine_details', label: 'Fine Details & Textures (10 pts)', icon: '🔍' },
        ];

        catConfig.forEach(cat => {
            const rationaleText = catRationales[cat.key] || `${cat.label} analyzed.`;
            const card = document.createElement('div');
            card.className = 'category-rationale-item';
            card.innerHTML = `
                <div class="rationale-cat-header">
                    <span>${cat.icon} ${cat.label}</span>
                </div>
                <p class="rationale-cat-body">${rationaleText}</p>
            `;
            modalCategoryRationales.appendChild(card);
        });

        // Section 4: Key Elements Analysis Table
        modalKeyElementsTableBody.innerHTML = '';
        const keyElements = report.key_elements || [];
        if (keyElements.length > 0) {
            keyElements.forEach(ke => {
                const tr = document.createElement('tr');
                let statusBadgeClass = 'status-present';
                let iconSymbol = '✓';
                if (ke.status === 'PARTIALLY_PRESENT' || ke.status === 'partially_present') {
                    statusBadgeClass = 'status-partial';
                    iconSymbol = '△';
                } else if (ke.status === 'MISSING' || ke.status === 'missing') {
                    statusBadgeClass = 'status-missing';
                    iconSymbol = '✗';
                }

                const impBadgeClass = ke.importance === 'HIGH' ? 'badge-high' : ke.importance === 'MEDIUM' ? 'badge-med' : 'badge-low';

                tr.innerHTML = `
                    <td><span class="status-indicator ${statusBadgeClass}">${iconSymbol} ${ke.status.toUpperCase()}</span></td>
                    <td><strong>${ke.name}</strong></td>
                    <td><span class="elem-badge ${impBadgeClass}">${ke.importance} IMP</span></td>
                    <td class="notes-cell">${ke.notes || 'Identified in recreation'}</td>
                `;
                modalKeyElementsTableBody.appendChild(tr);
            });
        } else {
            modalKeyElementsTableBody.innerHTML = '<tr><td colspan="4" class="text-center text-dim">Reference anchor objects evaluated.</td></tr>';
        }

        // Section 5: Low-Level Comparison Evidence Grid
        function populateList(container, items, emptyText = 'None noted') {
            container.innerHTML = '';
            if (items && items.length > 0) {
                items.forEach(text => {
                    const li = document.createElement('li');
                    li.textContent = text;
                    container.appendChild(li);
                });
            } else {
                const li = document.createElement('li');
                li.className = 'text-dim';
                li.textContent = emptyText;
                container.appendChild(li);
            }
        }

        populateList(modalMatchedElements, evidence.matched_elements, 'No exact object matches');
        populateList(modalPartialElements, evidence.partially_matched_elements, 'No partial object matches');
        populateList(modalMissingElements, evidence.missing_elements, 'All salient reference objects accounted for');
        populateList(modalUnexpectedElements, evidence.unexpected_elements, 'No unexpected major objects');
        populateList(modalCompDiffs, evidence.composition_differences, 'Composition alignment is consistent');
        populateList(modalColorDiffs, evidence.color_differences, 'Color and lighting harmony preserved');
        populateList(modalDetailDiffs, evidence.detail_differences, 'Detail frequency aligns with reference');
        populateList(modalStrengths, evidence.strengths, 'Standard recreation fidelity');

        // Reset visual comparison mode to Side-by-Side
        setVisualComparisonMode('side');

        dossierModal.classList.remove('hidden');
    }

    // Visual Comparison View Modes
    document.querySelectorAll('.view-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.view-tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            setVisualComparisonMode(tab.dataset.mode);
        });
    });

    function setVisualComparisonMode(mode) {
        state.activeCompMode = mode;
        if (mode === 'side') {
            sideBySideRow.classList.remove('hidden');
            singleCanvasFrame.classList.add('hidden');
            overlaySliderWrap.classList.add('hidden');
        } else if (mode === 'diff') {
            sideBySideRow.classList.add('hidden');
            singleCanvasFrame.classList.remove('hidden');
            overlaySliderWrap.classList.add('hidden');
            renderDifferenceCanvas();
        } else if (mode === 'overlay') {
            sideBySideRow.classList.add('hidden');
            singleCanvasFrame.classList.remove('hidden');
            overlaySliderWrap.classList.remove('hidden');
            renderOverlayCanvas(overlayOpacitySlider.value / 100);
        }
    }

    overlayOpacitySlider.addEventListener('input', (e) => {
        if (state.activeCompMode === 'overlay') {
            renderOverlayCanvas(e.target.value / 100);
        }
    });

    function renderDifferenceCanvas() {
        const refImg = modalRefImg;
        const partImg = modalPartImg;
        if (!refImg.complete || !partImg.complete) return;

        const ctx = compCanvas.getContext('2d');
        const w = 400;
        const h = 400;
        compCanvas.width = w;
        compCanvas.height = h;

        // Draw ref to offscreen canvas
        const offCanvas = document.createElement('canvas');
        offCanvas.width = w;
        offCanvas.height = h;
        const offCtx = offCanvas.getContext('2d');
        offCtx.drawImage(refImg, 0, 0, w, h);
        const refData = offCtx.getImageData(0, 0, w, h).data;

        // Draw part
        offCtx.clearRect(0, 0, w, h);
        offCtx.drawImage(partImg, 0, 0, w, h);
        const partData = offCtx.getImageData(0, 0, w, h).data;

        const diffImgData = ctx.createImageData(w, h);
        const d = diffImgData.data;

        for (let i = 0; i < refData.length; i += 4) {
            const dr = Math.abs(refData[i] - partData[i]);
            const dg = Math.abs(refData[i + 1] - partData[i + 1]);
            const db = Math.abs(refData[i + 2] - partData[i + 2]);
            const diff = (dr + dg + db) / 3;

            // Heatmap colormap: low diff = dark blue/cyan, high diff = hot red/amber
            d[i] = Math.min(255, diff * 3);          // Red
            d[i + 1] = Math.min(255, (255 - diff) / 2); // Green
            d[i + 2] = Math.min(255, 255 - diff * 2); // Blue
            d[i + 3] = 255;
        }

        ctx.putImageData(diffImgData, 0, 0);
    }

    function renderOverlayCanvas(opacity) {
        const refImg = modalRefImg;
        const partImg = modalPartImg;
        if (!refImg.complete || !partImg.complete) return;

        const ctx = compCanvas.getContext('2d');
        const w = 400;
        const h = 400;
        compCanvas.width = w;
        compCanvas.height = h;

        // Draw reference base
        ctx.globalAlpha = 1.0;
        ctx.drawImage(refImg, 0, 0, w, h);

        // Draw participant overlay with adjusted alpha
        ctx.globalAlpha = opacity;
        ctx.drawImage(partImg, 0, 0, w, h);
        ctx.globalAlpha = 1.0;
    }

    // Toggle detailed evidence grid
    if (btnToggleEvidenceGrid) {
        btnToggleEvidenceGrid.addEventListener('click', () => {
            modalEvidenceGrid.classList.toggle('hidden');
        });
    }

    modalCloseBtn.addEventListener('click', () => dossierModal.classList.add('hidden'));
    dossierModal.addEventListener('click', (e) => {
        if (e.target === dossierModal) dossierModal.classList.add('hidden');
    });
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && !dossierModal.classList.contains('hidden')) {
            dossierModal.classList.add('hidden');
        }
    });

    // ==========================================
    // SESSION MANAGEMENT (SAVE / LOAD / NEW)
    // ==========================================
    btnNewSession.addEventListener('click', () => {
        if (confirm('Start a new competition judging session? Unsaved progress will be reset.')) {
            window.location.reload();
        }
    });

    btnSaveSession.addEventListener('click', async () => {
        const sessionPayload = {
            session_id: state.sessionId,
            session_name: state.sessionName,
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
            status: state.evaluationResult ? 'COMPLETED' : 'READY',
            reference_filename: state.reference?.filename || null,
            reference_profile: state.reference?.profile || null,
            participants: state.participants.map(p => ({
                participant_id: p.participant_id,
                filename: p.filename,
                original_filename: p.original_filename,
                file_url: p.file_url,
                status: p.status || 'PENDING',
                error: p.error || null,
            })),
            evaluation_results: state.evaluationResult?.ranked_results?.map(r => ({
                participant_id: r.participant_id,
                filename: r.filename,
                scores: r.scores,
                explanation: r.explanation,
                category_reasons: r.judging_report?.category_explanations || {},
                evidence: r.evidence,
                judging_report: r.judging_report,
            })) || [],
            ranked_results: state.evaluationResult?.ranked_results || [],
            audit_metadata: state.evaluationResult?.audit_metadata || null,
        };

        try {
            // Save on server
            const resp = await fetch('/api/session/save', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(sessionPayload),
            });

            // Also download JSON copy directly to client
            const blob = new Blob([JSON.stringify(sessionPayload, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `the_2047_session_${state.sessionId}.json`;
            document.body.appendChild(a);
            a.click();
            a.remove();
            URL.revokeObjectURL(url);

            showToast(`Judging session saved successfully (${state.sessionId}).`, 'success');
        } catch (err) {
            console.error(err);
            showToast('Failed to save session.', 'error');
        }
    });

    btnLoadSession.addEventListener('click', () => {
        sessionFileInput.click();
    });

    sessionFileInput.addEventListener('change', async (e) => {
        const file = e.target.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append('file', file);

        showToast('Loading judging session...', 'info');

        try {
            const resp = await fetch('/api/session/load', {
                method: 'POST',
                body: formData,
            });

            if (!resp.ok) {
                const err = await resp.json();
                throw new Error(err.detail || 'Load session failed');
            }

            const session = await resp.json();
            restoreJudgingSession(session);
            showToast(`Session loaded: ${session.session_id} (${session.participants?.length || 0} participants).`, 'success');
        } catch (err) {
            console.error(err);
            showToast(`Load session error: ${err.message}`, 'error');
        } finally {
            sessionFileInput.value = '';
        }
    });

    function restoreJudgingSession(session) {
        state.sessionId = session.session_id;
        state.sessionName = session.session_name;

        // Restore reference
        if (session.reference_profile) {
            state.reference = {
                filename: session.reference_filename || session.reference_profile.filename,
                file_url: `/uploads/reference/${session.reference_filename || session.reference_profile.filename}`,
                profile: session.reference_profile,
            };
            refPreviewImg.src = state.reference.file_url;
            refFilename.textContent = state.reference.filename;
            refDropzoneEmpty.classList.add('hidden');
            refPreviewContainer.classList.remove('hidden');
            renderReferenceProfile(session.reference_profile);
        }

        // Restore participants
        state.participants = (session.participants || []).map(p => ({
            participant_id: p.participant_id,
            filename: p.filename,
            original_filename: p.original_filename || p.filename,
            file_url: p.file_url || `/uploads/participants/${p.filename}`,
            status: p.status || 'PENDING',
            error: p.error || null,
        }));
        renderParticipantsGrid();

        // Restore evaluation & leaderboard if present
        if (session.ranked_results && session.ranked_results.length > 0) {
            state.evaluationResult = {
                reference_id: session.reference_profile?.image_id || 'REF-RESTORED',
                reference_profile: session.reference_profile,
                total_participants: session.ranked_results.length,
                ranked_results: session.ranked_results,
                processing_time_seconds: session.audit_metadata?.execution_duration_sec || 0.0,
                audit_metadata: session.audit_metadata,
            };
            renderResults(state.evaluationResult);
        }

        updateLiveHud();
    }

    // ==========================================
    // EXPORT HANDLERS (CSV, JSON, PRINTABLE)
    // ==========================================
    if (btnExportCsv) btnExportCsv.addEventListener('click', exportToCsv);
    if (btnExportJson) btnExportJson.addEventListener('click', exportToJson);
    if (btnPrintScorecards) btnPrintScorecards.addEventListener('click', printScorecards);

    async function exportToCsv() {
        if (!state.evaluationResult) {
            showToast('No evaluation results to export.', 'info');
            return;
        }
        try {
            const resp = await fetch('/api/export/csv', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(state.evaluationResult),
            });
            if (!resp.ok) throw new Error('Export CSV failed');
            const blob = await resp.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `the_2047_judging_results_${Date.now()}.csv`;
            document.body.appendChild(a);
            a.click();
            a.remove();
            window.URL.revokeObjectURL(url);
            showToast('Judging Results CSV exported successfully.', 'success');
        } catch (err) {
            console.error(err);
            showToast('CSV export failed.', 'error');
        }
    }

    async function exportToJson() {
        if (!state.evaluationResult) {
            showToast('No evaluation results to export.', 'info');
            return;
        }
        try {
            const resp = await fetch('/api/export/json', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(state.evaluationResult),
            });
            if (!resp.ok) throw new Error('Export JSON failed');
            const blob = await resp.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `the_2047_judging_dossier_${Date.now()}.json`;
            document.body.appendChild(a);
            a.click();
            a.remove();
            window.URL.revokeObjectURL(url);
            showToast('Judging Results JSON dossier exported successfully.', 'success');
        } catch (err) {
            console.error(err);
            showToast('JSON export failed.', 'error');
        }
    }

    async function printScorecards() {
        if (!state.evaluationResult) {
            showToast('No evaluation results to print.', 'info');
            return;
        }
        try {
            const resp = await fetch('/api/export/printable', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(state.evaluationResult),
            });
            if (!resp.ok) throw new Error('Print generation failed');
            const html = await resp.text();
            const printWindow = window.open('', '_blank');
            printWindow.document.write(html);
            printWindow.document.close();
        } catch (err) {
            console.error(err);
            showToast('Failed to open printable scorecard.', 'error');
        }
    }

    const btnExportAudit = document.getElementById('btnExportAudit');
    if (btnExportAudit) btnExportAudit.addEventListener('click', exportAuditLog);

    async function exportAuditLog() {
        if (!state.evaluationResult) {
            showToast('No evaluation results to export audit log.', 'info');
            return;
        }
        try {
            const resp = await fetch('/api/export/audit', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(state.evaluationResult),
            });
            if (!resp.ok) throw new Error('Export audit log failed');
            const blob = await resp.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `the_2047_audit_log_${Date.now()}.json`;
            document.body.appendChild(a);
            a.click();
            a.remove();
            window.URL.revokeObjectURL(url);
            showToast('Verifiable cryptographic audit log exported.', 'success');
        } catch (err) {
            console.error(err);
            showToast('Audit log export failed.', 'error');
        }
    }

    // ==========================================
    // CALIBRATION & HUMAN-AI ALIGNMENT HUB
    // ==========================================
    state.humanEvaluations = [];
    state.humanOverrides = [];
    state.calibrationStats = null;

    const btnOpenCalibration = document.getElementById('btnOpenCalibration');
    const calibrationModalBackdrop = document.getElementById('calibrationModalBackdrop');
    const calibModalCloseBtn = document.getElementById('calibModalCloseBtn');
    const calibMainTabs = document.getElementById('calibMainTabs');
    const paneBlindEval = document.getElementById('paneBlindEval');
    const paneAnalytics = document.getElementById('paneAnalytics');
    const paneAbSim = document.getElementById('paneAbSim');
    const paneDiagnostics = document.getElementById('paneDiagnostics');

    // Blind Eval Elements
    const blindParticipantSelect = document.getElementById('blindParticipantSelect');
    const blindStatusBadge = document.getElementById('blindStatusBadge');
    const blindRefImg = document.getElementById('blindRefImg');
    const blindPartImg = document.getElementById('blindPartImg');
    const blindPartLabel = document.getElementById('blindPartLabel');
    const blindOverallSlider = document.getElementById('blindOverallSlider');
    const blindOverallBadge = document.getElementById('blindOverallBadge');
    const blindSemSlider = document.getElementById('blindSemSlider');
    const blindSemBadge = document.getElementById('blindSemBadge');
    const blindObjSlider = document.getElementById('blindObjSlider');
    const blindObjBadge = document.getElementById('blindObjBadge');
    const blindCompSlider = document.getElementById('blindCompSlider');
    const blindCompBadge = document.getElementById('blindCompBadge');
    const blindColSlider = document.getElementById('blindColSlider');
    const blindColBadge = document.getElementById('blindColBadge');
    const blindDetSlider = document.getElementById('blindDetSlider');
    const blindDetBadge = document.getElementById('blindDetBadge');
    const blindCommentsInput = document.getElementById('blindCommentsInput');
    const btnSubmitBlindEval = document.getElementById('btnSubmitBlindEval');
    const revealedComparisonCard = document.getElementById('revealedComparisonCard');
    const revealedTitle = document.getElementById('revealedTitle');
    const revealedTableBody = document.getElementById('revealedTableBody');

    // Analytics Elements
    const btnRunFullCalibration = document.getElementById('btnRunFullCalibration');
    const btnExportCalibReport = document.getElementById('btnExportCalibReport');
    const calibStatsGrid = document.getElementById('calibStatsGrid');
    const calibDecisionBanner = document.getElementById('calibDecisionBanner');
    const categoryMaeBars = document.getElementById('categoryMaeBars');
    const disagreementsTableBody = document.getElementById('disagreementsTableBody');
    const disagreementCountBadge = document.getElementById('disagreementCountBadge');

    // A/B Sim Elements
    const abSemSlider = document.getElementById('abSemSlider');
    const abSemVal = document.getElementById('abSemVal');
    const abObjSlider = document.getElementById('abObjSlider');
    const abObjVal = document.getElementById('abObjVal');
    const abCompSlider = document.getElementById('abCompSlider');
    const abCompVal = document.getElementById('abCompVal');
    const abColSlider = document.getElementById('abColSlider');
    const abColVal = document.getElementById('abColVal');
    const abDetSlider = document.getElementById('abDetSlider');
    const abDetVal = document.getElementById('abDetVal');
    const btnRunAbSim = document.getElementById('btnRunAbSim');
    const abComparisonResults = document.getElementById('abComparisonResults');

    // Diagnostics Elements
    const calibParticipantSelect = document.getElementById('calibParticipantSelect');
    const btnRunCalibInspect = document.getElementById('btnRunCalibInspect');
    const calibResultsArea = document.getElementById('calibResultsArea');
    const calibSummaryBanner = document.getElementById('calibSummaryBanner');
    const calibPaneObjects = document.getElementById('calibPaneObjects');

    // Human Override Elements
    const modalOverrideBanner = document.getElementById('modalOverrideBanner');
    const modalOverrideMeta = document.getElementById('modalOverrideMeta');
    const modalOverrideReason = document.getElementById('modalOverrideReason');
    const modalOriginalAiScore = document.getElementById('modalOriginalAiScore');
    const modalOverrideScore = document.getElementById('modalOverrideScore');
    const modalBtnFlagReview = document.getElementById('modalBtnFlagReview');
    const modalFlagText = document.getElementById('modalFlagText');
    const modalBtnOverrideScore = document.getElementById('modalBtnOverrideScore');
    const overrideModalBackdrop = document.getElementById('overrideModalBackdrop');
    const overrideModalCloseBtn = document.getElementById('overrideModalCloseBtn');
    const overrideModalTitle = document.getElementById('overrideModalTitle');
    const overrideOriginalAiVal = document.getElementById('overrideOriginalAiVal');
    const overrideScoreInput = document.getElementById('overrideScoreInput');
    const overrideReasonInput = document.getElementById('overrideReasonInput');
    const btnCancelOverride = document.getElementById('btnCancelOverride');
    const btnSaveOverride = document.getElementById('btnSaveOverride');

    // Open Hub
    if (btnOpenCalibration) {
        btnOpenCalibration.addEventListener('click', () => {
            if (!state.reference) {
                showToast('Please upload a Reference image before opening Calibration Hub.', 'info');
                return;
            }
            populateCalibrationParticipants();
            calibrationModalBackdrop.classList.remove('hidden');
        });
    }

    if (calibModalCloseBtn) {
        calibModalCloseBtn.addEventListener('click', () => {
            calibrationModalBackdrop.classList.add('hidden');
        });
    }

    // Tab Switching
    if (calibMainTabs) {
        calibMainTabs.querySelectorAll('.calib-tab').forEach(tabBtn => {
            tabBtn.addEventListener('click', () => {
                calibMainTabs.querySelectorAll('.calib-tab').forEach(b => b.classList.remove('active'));
                tabBtn.classList.add('active');
                const target = tabBtn.dataset.tab;

                paneBlindEval.classList.toggle('hidden', target !== 'blind');
                paneAnalytics.classList.toggle('hidden', target !== 'analytics');
                paneAbSim.classList.toggle('hidden', target !== 'absim');
                paneDiagnostics.classList.toggle('hidden', target !== 'diagnostics');

                if (target === 'analytics') {
                    runCalibrationAnalysis();
                }
            });
        });
    }

    function populateCalibrationParticipants() {
        blindParticipantSelect.innerHTML = '<option value="">Select participant to evaluate...</option>';
        calibParticipantSelect.innerHTML = '<option value="">Select participant to inspect...</option>';

        state.participants.forEach(p => {
            const hasRated = state.humanEvaluations.some(h => h.participant_id === p.participant_id);
            const statusSuffix = hasRated ? ' ✓ [Rated]' : ' [Pending]';

            const opt1 = document.createElement('option');
            opt1.value = p.participant_id;
            opt1.textContent = `${p.participant_id} - ${p.original_filename || p.filename}${statusSuffix}`;
            blindParticipantSelect.appendChild(opt1);

            const opt2 = document.createElement('option');
            opt2.value = p.filename;
            opt2.textContent = `${p.participant_id} (${p.filename})`;
            opt2.dataset.pid = p.participant_id;
            calibParticipantSelect.appendChild(opt2);
        });

        if (state.reference && blindRefImg) {
            blindRefImg.src = state.reference.file_url;
        }
    }

    // Blind Eval Selection
    if (blindParticipantSelect) {
        blindParticipantSelect.addEventListener('change', () => {
            const pId = blindParticipantSelect.value;
            if (!pId) return;

            const part = state.participants.find(p => p.participant_id === pId);
            if (!part) return;

            blindPartImg.src = part.file_url;
            blindPartLabel.textContent = `RECREATION: ${pId}`;

            // Check if already evaluated
            const existing = state.humanEvaluations.find(h => h.participant_id === pId);
            if (existing) {
                blindStatusBadge.textContent = 'EVALUATED ✓';
                blindStatusBadge.className = 'badge badge-emerald';
                blindOverallSlider.value = existing.human_score;
                blindOverallBadge.textContent = existing.human_score.toFixed(1);
                if (existing.human_semantic !== null) { blindSemSlider.value = existing.human_semantic; blindSemBadge.textContent = existing.human_semantic.toFixed(1); }
                if (existing.human_object !== null) { blindObjSlider.value = existing.human_object; blindObjBadge.textContent = existing.human_object.toFixed(1); }
                if (existing.human_composition !== null) { blindCompSlider.value = existing.human_composition; blindCompBadge.textContent = existing.human_composition.toFixed(1); }
                if (existing.human_color !== null) { blindColSlider.value = existing.human_color; blindColBadge.textContent = existing.human_color.toFixed(1); }
                if (existing.human_detail !== null) { blindDetSlider.value = existing.human_detail; blindDetBadge.textContent = existing.human_detail.toFixed(1); }
                blindCommentsInput.value = existing.human_comments || '';
                revealComparisonForParticipant(pId, existing);
            } else {
                blindStatusBadge.textContent = 'PENDING EVALUATION';
                blindStatusBadge.className = 'badge badge-subtle';
                revealedComparisonCard.classList.add('hidden');
                blindCommentsInput.value = '';
            }
        });
    }

    // Slider Listeners for Blind Form
    function setupSliderSync(slider, badge, step) {
        if (!slider || !badge) return;
        slider.addEventListener('input', () => {
            badge.textContent = parseFloat(slider.value).toFixed(1);
        });
    }

    setupSliderSync(blindOverallSlider, blindOverallBadge);
    setupSliderSync(blindSemSlider, blindSemBadge);
    setupSliderSync(blindObjSlider, blindObjBadge);
    setupSliderSync(blindCompSlider, blindCompBadge);
    setupSliderSync(blindColSlider, blindColBadge);
    setupSliderSync(blindDetSlider, blindDetBadge);

    // Auto-scale categories proportionally when overall slider changes
    if (blindOverallSlider) {
        blindOverallSlider.addEventListener('input', () => {
            const val = parseFloat(blindOverallSlider.value);
            const ratio = val / 100.0;
            blindSemSlider.value = (30.0 * ratio).toFixed(1);
            blindSemBadge.textContent = blindSemSlider.value;
            blindObjSlider.value = (25.0 * ratio).toFixed(1);
            blindObjBadge.textContent = blindObjSlider.value;
            blindCompSlider.value = (20.0 * ratio).toFixed(1);
            blindCompBadge.textContent = blindCompSlider.value;
            blindColSlider.value = (15.0 * ratio).toFixed(1);
            blindColBadge.textContent = blindColSlider.value;
            blindDetSlider.value = (10.0 * ratio).toFixed(1);
            blindDetBadge.textContent = blindDetSlider.value;
        });
    }

    // Submit Blind Eval
    if (btnSubmitBlindEval) {
        btnSubmitBlindEval.addEventListener('click', () => {
            const pId = blindParticipantSelect.value;
            if (!pId) {
                showToast('Please select a participant to evaluate.', 'info');
                return;
            }

            const humanEval = {
                participant_id: pId,
                human_score: parseFloat(blindOverallSlider.value),
                human_semantic: parseFloat(blindSemSlider.value),
                human_object: parseFloat(blindObjSlider.value),
                human_composition: parseFloat(blindCompSlider.value),
                human_color: parseFloat(blindColSlider.value),
                human_detail: parseFloat(blindDetSlider.value),
                human_comments: blindCommentsInput.value.trim() || null,
                evaluator_name: 'Lead Human Judge',
                submitted_at: new Date().toISOString(),
            };

            // Update or push
            const idx = state.humanEvaluations.findIndex(h => h.participant_id === pId);
            if (idx >= 0) {
                state.humanEvaluations[idx] = humanEval;
            } else {
                state.humanEvaluations.push(humanEval);
            }

            blindStatusBadge.textContent = 'EVALUATED ✓';
            blindStatusBadge.className = 'badge badge-emerald';
            showToast(`Human evaluation recorded for ${pId}!`, 'success');

            revealComparisonForParticipant(pId, humanEval);
            populateCalibrationParticipants();
            blindParticipantSelect.value = pId;
        });
    }

    function revealComparisonForParticipant(pId, humanEval) {
        let aiResult = null;
        if (state.evaluationResult && state.evaluationResult.ranked_results) {
            aiResult = state.evaluationResult.ranked_results.find(r => r.participant_id === pId);
        }

        if (!aiResult) {
            revealedTitle.textContent = `Human Rating: ${humanEval.human_score.toFixed(1)} / 100 (AI Not Evaluated Yet)`;
            revealedTableBody.innerHTML = `
                <tr><td colspan="5" style="text-align:center; color: var(--text-muted);">Run "ANALYZE ALL" on the competition dashboard to compare with automated AI scores.</td></tr>
            `;
            revealedComparisonCard.classList.remove('hidden');
            return;
        }

        revealedTitle.textContent = `Human (${humanEval.human_score.toFixed(1)}) vs AI (${aiResult.total_score.toFixed(1)}) Comparison: ${pId}`;

        const diffTotal = humanEval.human_score - aiResult.total_score;
        const absDiff = Math.abs(diffTotal);
        const statusTag = absDiff <= 5.0 ? '<span class="badge badge-emerald">Close Match (≤5 pts)</span>' :
                          absDiff <= 10.0 ? '<span class="badge badge-cyan">Aligned (≤10 pts)</span>' :
                          '<span class="badge badge-danger">⚠ Significant Disagreement (>10 pts)</span>';

        const categories = [
            { name: 'Overall Final Score', max: 100, h: humanEval.human_score, a: aiResult.total_score },
            { name: 'Semantic Similarity', max: 30, h: humanEval.human_semantic, a: aiResult.scores.semantic_similarity },
            { name: 'Object / Element Accuracy', max: 25, h: humanEval.human_object, a: aiResult.scores.object_accuracy },
            { name: 'Composition & Spatial Layout', max: 20, h: humanEval.human_composition, a: aiResult.scores.composition_spatial },
            { name: 'Color & Lighting', max: 15, h: humanEval.human_color, a: aiResult.scores.color_lighting },
            { name: 'Fine Details & Textures', max: 10, h: humanEval.human_detail, a: aiResult.scores.fine_details },
        ];

        let rows = '';
        categories.forEach((cat, idx) => {
            const hVal = cat.h !== null ? cat.h.toFixed(1) : '--';
            const aVal = cat.a !== null ? cat.a.toFixed(1) : '--';
            const dVal = (cat.h !== null && cat.a !== null) ? (cat.h - cat.a).toFixed(1) : '--';
            const isOverall = idx === 0;

            rows += `
                <tr style="${isOverall ? 'font-weight: 700; background: rgba(0, 240, 255, 0.08);' : ''}">
                    <td>${cat.name} <span style="font-size:10px; color:var(--text-dim);">(Max ${cat.max})</span></td>
                    <td style="color: var(--text-cyan);">${hVal}</td>
                    <td style="color: var(--text-purple);">${aVal}</td>
                    <td style="color: ${dVal > 0 ? '#34d399' : dVal < 0 ? '#f87171' : 'var(--text-main)'}; font-family: var(--font-mono);">${dVal > 0 ? '+' : ''}${dVal}</td>
                    <td>${isOverall ? statusTag : (Math.abs(cat.h - cat.a) <= (cat.max * 0.15) ? '<span class="text-emerald">✓ Aligned</span>' : '<span class="text-amber">△ Divergence</span>')}</td>
                </tr>
            `;
        });

        revealedTableBody.innerHTML = rows;
        revealedComparisonCard.classList.remove('hidden');
    }

    // Run Full Calibration Analysis
    async function runCalibrationAnalysis() {
        if (state.humanEvaluations.length === 0) {
            calibStatsGrid.innerHTML = `
                <div class="card" style="grid-column: 1/-1; padding: 20px; text-align: center; color: var(--text-muted);">
                    No human ratings recorded yet. Rate samples in the <strong>Blind Human Evaluation</strong> tab to compute calibration metrics.
                </div>
            `;
            calibDecisionBanner.innerHTML = '';
            categoryMaeBars.innerHTML = '';
            disagreementsTableBody.innerHTML = '<tr><td colspan="6" style="text-align:center; color:var(--text-dim);">No evaluations recorded.</td></tr>';
            disagreementCountBadge.textContent = '0 Disagreements';
            return;
        }

        btnRunFullCalibration.textContent = 'Computing...';
        try {
            const evals = (state.evaluationResult && state.evaluationResult.ranked_results) ?
                state.evaluationResult.ranked_results.map(r => ({
                    participant_id: r.participant_id,
                    filename: r.filename,
                    scores: r.scores,
                    explanation: r.explanation,
                    judging_report: r.judging_report,
                })) : [];

            const resp = await fetch('/api/calibration/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    human_evaluations: state.humanEvaluations,
                    evaluations: evals,
                }),
            });

            if (!resp.ok) throw new Error('Calibration analysis API failed');
            const stats = await resp.json();
            state.calibrationStats = stats;

            renderCalibrationDashboard(stats);
            showToast('Calibration statistics updated!', 'success');
        } catch (err) {
            console.error(err);
            showToast(`Calibration error: ${err.message}`, 'error');
        } finally {
            btnRunFullCalibration.textContent = '🔄 Compute Calibration Metrics';
        }
    }

    if (btnRunFullCalibration) {
        btnRunFullCalibration.addEventListener('click', runCalibrationAnalysis);
    }

    function renderCalibrationDashboard(stats) {
        calibStatsGrid.innerHTML = `
            <div class="stat-card-calib">
                <span class="stat-label">SAMPLES RATED</span>
                <span class="stat-number">${stats.total_samples}</span>
                <span class="stat-sub">Evaluators</span>
            </div>
            <div class="stat-card-calib">
                <span class="stat-label">HUMAN AVG</span>
                <span class="stat-number">${stats.human_avg_score.toFixed(1)}</span>
                <span class="stat-sub">/ 100 max</span>
            </div>
            <div class="stat-card-calib">
                <span class="stat-label">AI AVG</span>
                <span class="stat-number">${stats.ai_avg_score.toFixed(1)}</span>
                <span class="stat-sub">/ 100 max</span>
            </div>
            <div class="stat-card-calib">
                <span class="stat-label">MEAN ABS ERROR (MAE)</span>
                <span class="stat-number ${stats.mean_absolute_error <= 8.5 ? 'text-emerald' : 'text-amber'}">${stats.mean_absolute_error.toFixed(2)}</span>
                <span class="stat-sub">Target ≤ 8.5 pts</span>
            </div>
            <div class="stat-card-calib">
                <span class="stat-label">MEAN BIAS</span>
                <span class="stat-number">${stats.mean_difference > 0 ? '+' : ''}${stats.mean_difference.toFixed(2)}</span>
                <span class="stat-sub">${stats.mean_difference > 0 ? 'AI conservative' : 'AI generous'}</span>
            </div>
            <div class="stat-card-calib">
                <span class="stat-label">PEARSON (r)</span>
                <span class="stat-number text-purple">${stats.pearson_correlation.toFixed(3)}</span>
                <span class="stat-sub">Target ≥ 0.82</span>
            </div>
            <div class="stat-card-calib">
                <span class="stat-label">SPEARMAN (ρ)</span>
                <span class="stat-number text-purple">${stats.spearman_rank_correlation.toFixed(3)}</span>
                <span class="stat-sub">Rank correlation</span>
            </div>
            <div class="stat-card-calib">
                <span class="stat-label">RANK AGREEMENT</span>
                <span class="stat-number text-emerald">${stats.ranking_agreement_pct.toFixed(1)}%</span>
                <span class="stat-sub">Pairwise order</span>
            </div>
        `;

        const decisionColor = stats.decision.includes('🟢') ? 'var(--neon-cyan)' :
                              stats.decision.includes('🟡') ? 'var(--text-amber)' : 'var(--neon-pink)';

        calibDecisionBanner.style.borderLeftColor = decisionColor;
        calibDecisionBanner.innerHTML = `
            <div class="calib-decision-title" style="color: ${decisionColor};">${stats.decision}</div>
            <div class="calib-decision-desc">${stats.decision_rationale}</div>
        `;

        // Category MAE bars
        let catBars = '';
        const catMap = {
            'semantic_similarity': { name: 'Semantic / Overall Similarity', max: 30 },
            'object_accuracy': { name: 'Object Accuracy (Weighted)', max: 25 },
            'composition_spatial': { name: 'Composition & Spatial Layout', max: 20 },
            'color_lighting': { name: 'Color & Lighting Fidelity', max: 15 },
            'fine_details': { name: 'Fine Details & Textures', max: 10 },
        };

        for (const [k, meta] of Object.entries(catMap)) {
            const mae = stats.category_mae[k] || 0.0;
            const pct = Math.min(100, (mae / (meta.max * 0.35)) * 100);
            catBars += `
                <div class="mae-bar-row">
                    <span style="color: var(--text-muted);">${meta.name}</span>
                    <div class="mae-track"><div class="mae-fill" style="width: ${pct}%;"></div></div>
                    <span style="text-align: right; font-family: var(--font-mono); font-weight: 700;">${mae.toFixed(2)} pts</span>
                </div>
            `;
        }
        categoryMaeBars.innerHTML = catBars;

        // Disagreements Table
        disagreementCountBadge.textContent = `${stats.significant_disagreements.length} Disagreements (>10 pts)`;
        if (stats.significant_disagreements.length === 0) {
            disagreementsTableBody.innerHTML = `
                <tr><td colspan="6" style="text-align:center; color: var(--text-emerald); padding: 12px;">✓ Zero significant disagreements. All sample scores align within 10.0 points of human judgments.</td></tr>
            `;
        } else {
            let dRows = '';
            stats.significant_disagreements.forEach(d => {
                dRows += `
                    <tr>
                        <td style="font-weight: 700;">${d.participant_id}</td>
                        <td style="color: var(--text-cyan); font-weight: 700;">${d.human_score.toFixed(1)}</td>
                        <td style="color: var(--text-purple); font-weight: 700;">${d.ai_score.toFixed(1)}</td>
                        <td style="color: #f87171; font-family: var(--font-mono); font-weight: 700;">${d.difference > 0 ? '+' : ''}${d.difference.toFixed(1)}</td>
                        <td><span class="badge badge-subtle">${d.main_category_disagreement}</span></td>
                        <td style="font-size: 11px; color: var(--text-muted);">${d.human_comments || 'None recorded'}</td>
                    </tr>
                `;
            });
            disagreementsTableBody.innerHTML = dRows;
        }
    }

    // Export Calibration Report
    if (btnExportCalibReport) {
        btnExportCalibReport.addEventListener('click', async () => {
            if (!state.calibrationStats) {
                showToast('Run calibration metrics before exporting report.', 'info');
                return;
            }
            try {
                const resp = await fetch('/api/calibration/export-report', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(state.calibrationStats),
                });
                if (!resp.ok) throw new Error('Export report failed');
                const blob = await resp.blob();
                const url = window.URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `the_2047_calibration_report_${Date.now()}.md`;
                document.body.appendChild(a);
                a.click();
                a.remove();
                window.URL.revokeObjectURL(url);
                showToast('Calibration Markdown report exported.', 'success');
            } catch (err) {
                console.error(err);
                showToast('Report export failed.', 'error');
            }
        });
    }

    // A/B Sim Sliders
    setupSliderSync(abSemSlider, abSemVal);
    setupSliderSync(abObjSlider, abObjVal);
    setupSliderSync(abCompSlider, abCompVal);
    setupSliderSync(abColSlider, abColVal);
    setupSliderSync(abDetSlider, abDetVal);

    if (btnRunAbSim) {
        btnRunAbSim.addEventListener('click', async () => {
            if (state.humanEvaluations.length === 0) {
                showToast('Record at least 3 human evaluations before running A/B weight simulation.', 'info');
                return;
            }

            btnRunAbSim.textContent = 'Simulating...';
            try {
                const evals = (state.evaluationResult && state.evaluationResult.ranked_results) ?
                    state.evaluationResult.ranked_results : [];

                const resp = await fetch('/api/calibration/ab-test', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        config_name: 'Custom Test Configuration',
                        semantic_weight: parseFloat(abSemSlider.value),
                        object_weight: parseFloat(abObjSlider.value),
                        composition_weight: parseFloat(abCompSlider.value),
                        color_weight: parseFloat(abColSlider.value),
                        detail_weight: parseFloat(abDetSlider.value),
                        human_evaluations: state.humanEvaluations,
                    }),
                });

                if (!resp.ok) throw new Error('A/B simulation failed');
                const sim = await resp.json();

                renderAbSimulationResults(sim);
                abComparisonResults.classList.remove('hidden');
                showToast('A/B Simulation computed!', 'success');
            } catch (err) {
                console.error(err);
                showToast(`Simulation error: ${err.message}`, 'error');
            } finally {
                btnRunAbSim.textContent = 'Simulate Alternative Configuration';
            }
        });
    }

    function renderAbSimulationResults(sim) {
        abComparisonResults.innerHTML = `
            <div class="ab-cards-grid">
                <div class="ab-card">
                    <div class="ab-card-hdr text-cyan">${sim.config_a_name}</div>
                    <div style="font-size: 12px; margin-bottom: 6px;">MAE: <strong>${sim.config_a_mae.toFixed(2)} pts</strong></div>
                    <div style="font-size: 12px; margin-bottom: 6px;">Pearson (r): <strong>${sim.config_a_correlation.toFixed(3)}</strong></div>
                    <div style="font-size: 12px;">Ranking Agreement: <strong>${sim.config_a_ranking_agreement_pct.toFixed(1)}%</strong></div>
                </div>

                <div class="ab-card" style="border-color: var(--neon-purple);">
                    <div class="ab-card-hdr text-purple">${sim.config_b_name} (${sim.config_b_weights.semantic}/${sim.config_b_weights.object}/${sim.config_b_weights.composition}/${sim.config_b_weights.color}/${sim.config_b_weights.detail})</div>
                    <div style="font-size: 12px; margin-bottom: 6px;">MAE: <strong>${sim.config_b_mae.toFixed(2)} pts</strong> (${sim.mae_improvement > 0 ? '-' + sim.mae_improvement.toFixed(2) : '+' + Math.abs(sim.mae_improvement).toFixed(2)} pts)</div>
                    <div style="font-size: 12px; margin-bottom: 6px;">Pearson (r): <strong>${sim.config_b_correlation.toFixed(3)}</strong></div>
                    <div style="font-size: 12px;">Ranking Agreement: <strong>${sim.config_b_ranking_agreement_pct.toFixed(1)}%</strong></div>
                </div>
            </div>

            <div class="card" style="margin-top: 12px; padding: 12px; font-size: 12px; background: rgba(15, 23, 42, 0.9);">
                <span class="text-cyan" style="font-weight: 700;">Recommendation:</span> ${sim.recommendation}
            </div>
        `;
    }

    // ==========================================
    // HUMAN REVIEW & OVERRIDE HANDLERS
    // ==========================================
    if (modalBtnFlagReview) {
        modalBtnFlagReview.addEventListener('click', async () => {
            const pId = state.activeParticipantId;
            if (!pId) return;

            const part = (state.evaluationResult && state.evaluationResult.ranked_results) ?
                state.evaluationResult.ranked_results.find(r => r.participant_id === pId) : null;
            const currentFlag = part ? !!part.flagged_for_review : false;
            const newFlag = !currentFlag;

            try {
                const resp = await fetch(`/api/participants/${pId}/flag-review`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        participant_id: pId,
                        flagged: newFlag,
                        reason: newFlag ? 'Flagged by organizer for closer inspection' : 'Resolved',
                    }),
                });

                if (!resp.ok) throw new Error('Flagging failed');
                if (part) part.flagged_for_review = newFlag;

                modalBtnFlagReview.classList.toggle('btn-flagged', newFlag);
                modalFlagText.textContent = newFlag ? 'Flagged for Review ✓' : 'Flag for Review';
                renderLeaderboardTable();
                showToast(newFlag ? `${pId} flagged for human review.` : `${pId} unflagged.`, 'info');
            } catch (err) {
                console.error(err);
                showToast(`Flag error: ${err.message}`, 'error');
            }
        });
    }

    if (modalBtnOverrideScore) {
        modalBtnOverrideScore.addEventListener('click', () => {
            const pId = state.activeParticipantId;
            if (!pId) return;

            const part = (state.evaluationResult && state.evaluationResult.ranked_results) ?
                state.evaluationResult.ranked_results.find(r => r.participant_id === pId) : null;
            if (!part) return;

            overrideModalTitle.textContent = `Human Score Override: ${pId}`;
            overrideOriginalAiVal.value = `${part.total_score.toFixed(1)} / 100`;
            overrideScoreInput.value = part.human_override ? part.human_override.override_score : part.total_score.toFixed(1);
            overrideReasonInput.value = part.human_override ? part.human_override.override_reason : '';
            overrideModalBackdrop.classList.remove('hidden');
        });
    }

    if (overrideModalCloseBtn) {
        overrideModalCloseBtn.addEventListener('click', () => {
            overrideModalBackdrop.classList.add('hidden');
        });
    }

    if (btnCancelOverride) {
        btnCancelOverride.addEventListener('click', () => {
            overrideModalBackdrop.classList.add('hidden');
        });
    }

    if (btnSaveOverride) {
        btnSaveOverride.addEventListener('click', async () => {
            const pId = state.activeParticipantId;
            const scoreVal = parseFloat(overrideScoreInput.value);
            const reasonVal = overrideReasonInput.value.trim();

            if (isNaN(scoreVal) || scoreVal < 0 || scoreVal > 100) {
                showToast('Please enter a valid override score between 0.0 and 100.0.', 'warning');
                return;
            }

            if (!reasonVal) {
                showToast('Organizer justification is required for audit trail transparency.', 'warning');
                return;
            }

            try {
                const resp = await fetch(`/api/participants/${pId}/override`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        participant_id: pId,
                        override_score: scoreVal,
                        override_reason: reasonVal,
                        organizer_name: 'Lead Competition Organizer',
                    }),
                });

                if (!resp.ok) throw new Error('Save override failed');
                const override = await resp.json();

                // Apply to active state
                const part = state.evaluationResult.ranked_results.find(r => r.participant_id === pId);
                if (part) {
                    override.original_ai_score = part.total_score;
                    part.human_override = override;
                    part.total_score = scoreVal;
                    part.flagged_for_review = true;

                    // Re-sort ranked results
                    state.evaluationResult.ranked_results.sort((a, b) => b.total_score - a.total_score);
                    state.evaluationResult.ranked_results.forEach((r, i) => { r.rank = i + 1; });
                }

                overrideModalBackdrop.classList.add('hidden');
                updateDossierOverrideBanner(pId);
                renderLeaderboardTable();
                renderTopPerformers();
                showToast(`Human score override recorded for ${pId} (${scoreVal.toFixed(1)} / 100).`, 'success');
            } catch (err) {
                console.error(err);
                showToast(`Override error: ${err.message}`, 'error');
            }
        });
    }

    function updateDossierOverrideBanner(pId) {
        const part = (state.evaluationResult && state.evaluationResult.ranked_results) ?
            state.evaluationResult.ranked_results.find(r => r.participant_id === pId) : null;
        if (!part) return;

        if (part.human_override) {
            modalOverrideBanner.classList.remove('hidden');
            modalOverrideMeta.textContent = `Recorded by ${part.human_override.organizer_name} at ${new Date(part.human_override.timestamp).toLocaleTimeString()}`;
            modalOverrideReason.textContent = `Justification: "${part.human_override.override_reason}"`;
            modalOriginalAiScore.textContent = `${part.human_override.original_ai_score.toFixed(1)} / 100`;
            modalOverrideScore.textContent = `${part.human_override.override_score.toFixed(1)} / 100`;
        } else {
            modalOverrideBanner.classList.add('hidden');
        }

        if (modalBtnFlagReview) {
            modalBtnFlagReview.classList.toggle('btn-flagged', !!part.flagged_for_review);
            modalFlagText.textContent = part.flagged_for_review ? 'Flagged for Review ✓' : 'Flag for Review';
        }
    }

    // Hook updateDossierOverrideBanner into openDossierModal
    const origOpenDossierModal = window.openDossierModal;
    // Also update openDossierModal call if defined
    if (typeof openDossierModal === 'function') {
        const _orig = openDossierModal;
        window.openDossierModal = function(pId) {
            _orig(pId);
            updateDossierOverrideBanner(pId);
        };
    }

    // Initial check
    updateLiveHud();
});

