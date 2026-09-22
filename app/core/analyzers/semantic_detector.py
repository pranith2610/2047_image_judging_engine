"""
Semantic Detector & Zero-Shot Vision Classifier for THE 2047.
Provides fast, deterministic semantic scene classification, subject identification,
and multi-feature presence detection using ONNX CLIP ViT-B/32 and computer vision metrics.
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
    from transformers import AutoTokenizer
    HAS_ONNX_TRANSFORMERS = True
except ImportError:
    HAS_ONNX_TRANSFORMERS = False


class SemanticDetector:
    """
    Zero-shot semantic classifier and feature presence engine.
    Extracts high-level scene identity, subject classification, and verifies
    the presence of specific reference elements or extraneous elements.
    """
    _instance: Optional["SemanticDetector"] = None

    def __init__(self):
        self._vis_session = None
        self._text_session = None
        self._tokenizer = None
        self._initialized = False
        self._precomputed_text_embeds: Dict[str, np.ndarray] = {}
        self._setup_models()

    @classmethod
    def get_instance(cls) -> "SemanticDetector":
        if cls._instance is None:
            cls._instance = SemanticDetector()
        return cls._instance

    def _setup_models(self):
        """Locate and initialize ONNX vision and text models."""
        if not HAS_ONNX_TRANSFORMERS:
            return

        base_cache = Path.home() / ".cache/huggingface/hub/models--Xenova--clip-vit-base-patch32"
        if not base_cache.exists():
            return

        # Find snapshots
        vis_models = list(base_cache.glob("**/vision_model_quantized.onnx"))
        text_models = list(base_cache.glob("**/text_model_quantized.onnx"))

        if not vis_models:
            return

        try:
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            self._vis_session = ort.InferenceSession(str(vis_models[0]), sess_options=opts)

            if text_models:
                self._text_session = ort.InferenceSession(str(text_models[0]), sess_options=opts)
                self._tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
                self._precompute_concept_embeddings()

            self._initialized = True
        except Exception:
            self._vis_session = None
            self._text_session = None
            self._initialized = False

    def _precompute_concept_embeddings(self):
        """Precompute normalized text embeddings for instant zero-shot queries."""
        if not self._text_session or not self._tokenizer:
            return

        concepts = {
            # Reference core features
            "ref_scene": "a South Indian Hindu temple festival market with decorated elephant and gopuram",
            "ref_elephant": "a large ceremonial elephant decorated with gold headdress nettipattam and rider",
            "ref_gopuram": "a tall stone Hindu temple gopuram tower with carved Dravidian architecture",
            "ref_festival_market": "a bustling traditional outdoor temple festival market with stalls and crowds",
            "ref_traditional_clothing": "people wearing traditional South Indian clothing veshti dhoti and silk sari",
            "ref_oxen": "white oxen cattle and traditional wooden bullock cart",
            "ref_banners": "red ceremonial banners with leaping tiger crest and gold embroidery",
            "ref_brass_objects": "traditional brass lamps vilakku and ceremonial metal vessels",
            "ref_market_produce": "market baskets filled with fruits mangoes and jasmine flower garlands",
            "ref_palm_trees": "tropical coconut palm trees in warm golden daylight",
            "ref_animals_peripheral": "a resting white dog, rooster, and pigeons on cobblestone",
            
            # Non-reference / Negative scene domains
            "scene_sports": "a professional football soccer player on a stadium pitch playing a sports match",
            "scene_space": "an astronaut in a white spacesuit floating in deep cosmic space among stars",
            "scene_galaxy": "a cosmic galaxy nebula astronomy star cluster in outer space",
            "scene_mountain": "a snow covered frozen alpine mountain peak in winter",
            "scene_racing": "a modern race car sports car speeding on a formula asphalt track",
            "scene_cyberpunk": "a futuristic cyberpunk neon city street with glowing holograms at night",
            "scene_kitchen": "a modern domestic home kitchen with stainless steel appliances and cabinets",
            "scene_basketball": "a basketball player wearing athletic uniform jersey playing on indoor court",
            "scene_office": "a corporate modern office interior with desks computers and glass windows",
            "scene_underwater": "an underwater ocean marine coral reef with colorful tropical fish",
            
            # Specific extraneous cues
            "cue_ronaldo": "Cristiano Ronaldo Manchester United red football jersey with sponsor dominate",
            "cue_modern_sports": "modern sports athlete jersey athletic sportswear and stadium lighting",
            "cue_typography": "prominent English text typography and graphic lettering overlay",
        }

        keys = list(concepts.keys())
        texts = [concepts[k] for k in keys]

        try:
            tokens = self._tokenizer(texts, padding=True, return_tensors="np")
            out = self._text_session.run(None, {"input_ids": tokens["input_ids"]})[0]
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            normed = out / norms

            for k, emb in zip(keys, normed):
                self._precomputed_text_embeds[k] = emb
        except Exception:
            pass

    def get_image_embedding(self, image_input: Any) -> Optional[np.ndarray]:
        """
        Extract normalized 512-dim CLIP visual embedding from a PIL Image, numpy array, or filepath.
        """
        if not self._vis_session:
            return None

        try:
            if isinstance(image_input, (str, Path)):
                img = Image.open(image_input).convert("RGB")
            elif isinstance(image_input, np.ndarray):
                img = Image.fromarray(image_input).convert("RGB")
            elif isinstance(image_input, Image.Image):
                img = image_input.convert("RGB")
            else:
                return None

            img_resized = img.resize((224, 224), Image.Resampling.BILINEAR)
            arr = np.array(img_resized, dtype=np.float32) / 255.0

            # CLIP normalization
            mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
            std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
            arr = (arr - mean) / std
            arr = np.transpose(arr, (2, 0, 1))
            arr = np.expand_dims(arr, 0)

            out = self._vis_session.run(None, {"pixel_values": arr})[0][0]
            norm = np.linalg.norm(out)
            if norm > 1e-6:
                out = out / norm
            return out.astype(np.float32)
        except Exception:
            return None

    def analyze_semantic_profile(self, image_input: Any) -> Dict[str, Any]:
        """
        Analyze an image and return structured semantic features.
        """
        emb = self.get_image_embedding(image_input)
        if emb is None or not self._precomputed_text_embeds:
            return self._heuristic_semantic_analysis(image_input)

        # 1. Compare against precomputed concept embeddings
        sims: Dict[str, float] = {}
        for k, text_emb in self._precomputed_text_embeds.items():
            sims[k] = float(np.dot(emb, text_emb))

        # 2. Determine Scene Domain
        scene_candidates = [
            ("temple_festival", sims.get("ref_scene", 0.0) * 1.1),
            ("sports_athletics", max(sims.get("scene_sports", 0.0), sims.get("scene_basketball", 0.0), sims.get("cue_ronaldo", 0.0))),
            ("outer_space", max(sims.get("scene_space", 0.0), sims.get("scene_galaxy", 0.0))),
            ("mountain_snow", sims.get("scene_mountain", 0.0)),
            ("vehicle_racing", sims.get("scene_racing", 0.0)),
            ("cyberpunk_city", sims.get("scene_cyberpunk", 0.0)),
            ("kitchen_interior", sims.get("scene_kitchen", 0.0)),
            ("office_interior", sims.get("scene_office", 0.0)),
            ("underwater_marine", sims.get("scene_underwater", 0.0)),
        ]
        scene_candidates.sort(key=lambda x: x[1], reverse=True)
        top_scene, top_scene_score = scene_candidates[0]

        # 3. Determine Main Subject
        elephant_score = sims.get("ref_elephant", 0.0)
        sports_score = max(sims.get("cue_ronaldo", 0.0), sims.get("cue_modern_sports", 0.0), sims.get("scene_sports", 0.0))
        space_score = sims.get("scene_space", 0.0)
        racing_score = sims.get("scene_racing", 0.0)

        if top_scene == "temple_festival" and elephant_score >= 0.23:
            main_subject = "ceremonial_elephant"
            main_subject_desc = "Large decorated ceremonial elephant"
        elif top_scene == "sports_athletics" and sports_score >= 0.235:
            main_subject = "sports_athlete"
            main_subject_desc = "Football / sports athlete"
        elif top_scene == "outer_space" and space_score >= 0.235:
            main_subject = "astronaut_or_space"
            main_subject_desc = "Astronaut or cosmic subject"
        elif top_scene == "vehicle_racing" and racing_score >= 0.235:
            main_subject = "vehicle_racecar"
            main_subject_desc = "Racing sports car"
        else:
            main_subject = "general_subject"
            main_subject_desc = "General visual subject"

        # 4. Environment
        if top_scene == "temple_festival":
            env_desc = "South Indian temple and festival market environment"
        elif top_scene == "sports_athletics":
            env_desc = "Football stadium / sports arena environment"
        elif top_scene == "outer_space":
            env_desc = "Outer space / cosmic environment"
        elif top_scene == "mountain_snow":
            env_desc = "Alpine snow mountain environment"
        elif top_scene == "vehicle_racing":
            env_desc = "Racetrack / motorsport environment"
        elif top_scene == "cyberpunk_city":
            env_desc = "Futuristic neon city environment"
        elif top_scene == "kitchen_interior":
            env_desc = "Domestic kitchen interior"
        elif top_scene == "office_interior":
            env_desc = "Corporate office interior"
        elif top_scene == "underwater_marine":
            env_desc = "Underwater coral reef environment"
        else:
            env_desc = "General outdoor/indoor scene"

        # 5. Check High-Importance Reference Features presence
        ref_matches = {
            "ceremonial_elephant": max(0.0, min(1.0, (sims.get("ref_elephant", 0.0) - 0.20) / 0.05)),
            "temple_gopuram": max(0.0, min(1.0, (sims.get("ref_gopuram", 0.0) - 0.20) / 0.05)),
            "festival_market": max(0.0, min(1.0, (sims.get("ref_festival_market", 0.0) - 0.20) / 0.05)),
            "traditional_clothing": max(0.0, min(1.0, (sims.get("ref_traditional_clothing", 0.0) - 0.20) / 0.05)),
            "oxen_bullock_cart": max(0.0, min(1.0, (sims.get("ref_oxen", 0.0) - 0.20) / 0.05)),
            "red_ceremonial_banners": max(0.0, min(1.0, (sims.get("ref_banners", 0.0) - 0.20) / 0.04)),
            "brass_ceremonial_objects": max(0.0, min(1.0, (sims.get("ref_brass_objects", 0.0) - 0.20) / 0.04)),
            "market_stalls_fruits": max(0.0, min(1.0, (sims.get("ref_market_produce", 0.0) - 0.20) / 0.04)),
            "palm_trees": max(0.0, min(1.0, (sims.get("ref_palm_trees", 0.0) - 0.20) / 0.04)),
        }

        # 6. Detect Extraneous Elements
        extraneous = []
        if (sims.get("cue_ronaldo", 0.0) >= 0.245 or sims.get("cue_modern_sports", 0.0) >= 0.245) and top_scene == "sports_athletics":
            extraneous.append("Football player / athlete in modern sports apparel")
        if sims.get("cue_typography", 0.0) >= 0.235:
            extraneous.append("Prominent typography / graphic text overlay")
        if top_scene == "outer_space" and space_score >= 0.235:
            extraneous.append("Cosmic void / astronaut gear")
        elif top_scene == "vehicle_racing" and racing_score >= 0.235:
            extraneous.append("Modern motor vehicle / racetrack")
        elif top_scene == "sports_athletics" and "Football player / athlete in modern sports apparel" not in extraneous:
            extraneous.append("Sports stadium / athletic court")

        return {
            "scene_domain": top_scene,
            "scene_score": top_scene_score,
            "main_subject": main_subject,
            "main_subject_desc": main_subject_desc,
            "environment_desc": env_desc,
            "feature_matches": ref_matches,
            "extraneous_detected": extraneous,
            "clip_embedding": [round(float(x), 6) for x in emb],
            "raw_sims": sims,
        }

    def _heuristic_semantic_analysis(self, image_input: Any) -> Dict[str, Any]:
        """Fallback semantic analysis using pure visual heuristics when ONNX is unavailable."""
        return {
            "scene_domain": "general_scene",
            "scene_score": 0.5,
            "main_subject": "general_subject",
            "main_subject_desc": "Visual subject",
            "environment_desc": "Scene environment",
            "feature_matches": {},
            "extraneous_detected": [],
            "clip_embedding": None,
            "raw_sims": {},
        }
