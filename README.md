# THE 2047 — AI Image Comparison & Judging Engine

An AI-powered multi-dimensional visual judging and comparison platform engineered for image memorization & recreation competitions.

In **THE 2047**, participants observe a single reference image for 30 seconds without taking notes, then recreate the image from memory (or using generative AI tools). The engine evaluates up to 50 competitor recreations against the master reference image across **5 rigorous scoring categories totaling 100 points**, accounting for relative element importance (**HIGH**, **MEDIUM**, **LOW**).

---

## 🎯 Scoring Categories (100 Points Total)

| Category | Max Points | Evaluation Focus |
|---|---|---|
| **1. Semantic / Overall Visual Similarity** | **30 pts** | High-level scene theme, mood, visual atmosphere, and structural coherence. |
| **2. Object / Element Accuracy** | **25 pts** | Retention and placement of salient objects, weighted by relative importance (*HIGH* elements carry 60% of weight + deduction penalties if missed). |
| **3. Composition & Spatial Arrangement** | **20 pts** | Focal center alignment, 9-zone quadrant density correlation, rule of thirds, and horizontal symmetry. |
| **4. Color & Lighting** | **15 pts** | Dominant color palette fidelity (k-means clustering), average brightness, contrast ratio, and color temperature (warmth). |
| **5. Fine Details & Textures** | **10 pts** | Edge frequency (Canny/Sobel density), surface texture complexity, and Laplacian sharpness. |

---

## 🏗️ Core Architecture & Pipeline

```
REFERENCE IMAGE (30s memorization)
       ↓
REFERENCE ANALYZER ──→ Builds structured `ReferenceProfile` (Salient elements, HIGH/MED/LOW importance, palette, layout)
       ↓
PARTICIPANT IMAGE ANALYZER ──→ Extracts `ParticipantProfile` for up to 50 competitors
       ↓
FEATURE COMPARATOR ──→ Normalized multi-metric delta computations
       ↓
WEIGHTED SCORING ENGINE ──→ Evaluates exact 100-point category breakdowns with importance weights
       ↓
EXPLANATION ENGINE ──→ Natural language rationale explaining deductions and awards
       ↓
RANKING ENGINE ──→ Multi-tier tie-breaking, percentile calculation, and leaderboard generation
```

---

## 📁 Project Structure

```
├── app/
│   ├── config.py                # System settings, batch limits, upload paths
│   ├── main.py                  # FastAPI application entrypoint & static mounting
│   ├── api/
│   │   └── routes.py            # Upload, inspect, evaluate endpoints
│   ├── models/
│   │   ├── enums.py             # ImportanceLevel, ScoreCategory, SpatialQuadrant
│   │   └── schemas.py           # Pydantic v2 data models for Profiles, Scores, Rankings
│   ├── core/
│   │   ├── validator.py         # Format, dimension, size, and corruption validator
│   │   ├── coordinator.py       # Pipeline orchestrator
│   │   ├── analyzers/
│   │   │   ├── base.py          # Abstract BaseReferenceAnalyzer, BaseParticipantAnalyzer
│   │   │   ├── reference.py     # Reference profile computer vision analyzer
│   │   │   └── participant.py   # Participant feature extraction analyzer
│   │   ├── comparators/
│   │   │   ├── base.py          # Abstract BaseFeatureComparator
│   │   │   └── comparator.py    # Metric comparator (palette, layout, objects, details)
│   │   ├── scoring/
│   │   │   ├── base.py          # Abstract BaseScoringEngine
│   │   │   └── engine.py        # Weighted 100-pt scoring with element importance
│   │   ├── explanations/
│   │   │   ├── base.py          # Abstract BaseExplanationEngine
│   │   │   └── engine.py        # Natural language verdict generator
│   │   └── ranking/
│   │       ├── base.py          # Abstract BaseRankingEngine
│   │       └── engine.py        # Leaderboard sorting & tie-breaking
│   ├── static/
│   │   ├── css/styles.css       # THE 2047 Cyber/HUD dark glassmorphism design
│   │   └── js/app.js            # Batch upload manager, execution dock, modal inspection
│   └── templates/
│       └── index.html           # Full competition dashboard UI
├── tests/
│   ├── test_validation.py       # Image validation unit tests
│   ├── test_scoring.py          # 100-point invariant and importance penalty tests
│   └── test_pipeline.py         # End-to-end multi-image batch evaluation test
├── uploads/                     # Local session uploads (reference & participants)
├── run.py                       # CLI startup script
└── requirements.txt             # Project dependencies
```

---

## 🚀 Quick Start

### 1. Requirements
Ensure Python 3.10+ is installed with the required packages:
```bash
pip install -r requirements.txt
```
*(All core libraries like `fastapi`, `uvicorn`, `pillow`, `opencv-contrib-python`, `pydantic` are already verified in the local environment).*

### 2. Launch the Web Application
Run the launcher script:
```bash
python run.py
```
Or directly with Uvicorn:
```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser and navigate to:
```
http://127.0.0.1:8000
```

---

## 🧪 Running Automated Tests

Run the full automated test suite with pytest:
```bash
python -m pytest tests/ -v
```

---

## 🌐 Cloud Deployment (1-Click & Free Tier)

THE 2047 is configured for seamless deployment on free-tier Python hosting platforms:

### Option 1: Deploy on Render.com (Recommended)
1. Go to [Render.com](https://render.com) and create a free account.
2. Click **New +** → **Web Service** → Connect your GitHub repository (`pranith2610/2047_image_judging_engine`).
3. Render will automatically detect `render.yaml` and set:
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. Click **Deploy Web Service** — your live URL will be active in minutes!

### Option 2: Deploy on Railway.app
1. Go to [Railway.app](https://railway.app) and sign in with GitHub.
2. Click **New Project** → **Deploy from GitHub repo** → select `2047_image_judging_engine`.
3. Railway automatically detects the `Procfile` and launches the application.

### Option 3: Deploy with Docker
```bash
docker build -t the-2047-engine .
docker run -p 8000:8000 the-2047-engine
```

---

## 🔌 Extending With Deep Learning Models

Every component inherits from an abstract base class (`abc.ABC`). In future iterations:
- **CLIP ViT Embeddings**: Create `ClipReferenceAnalyzer` and `ClipComparator` implementing `BaseReferenceAnalyzer` and `BaseFeatureComparator` to calculate cosine similarity on visual embeddings.
- **YOLOv8 / Object Detection**: Plug into `BaseReferenceAnalyzer._identify_elements` to automatically recognize 80+ object classes and assign importance by bounding box area and centrality.
- **Vision-Language LLMs**: Implement `BaseExplanationEngine` to generate natural language explanations using local models or vision LLMs.
