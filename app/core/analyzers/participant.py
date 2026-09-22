"""
Participant image analyzer implementation.
Extracts visual features, color palettes, composition, visual embeddings, and details from participant recreations.
"""
from pathlib import Path
from typing import Optional, List
import numpy as np
from PIL import Image

from app.models.schemas import (
    ParticipantProfile,
    ReferenceProfile,
    StructureFeature,
    SceneFeature,
    ObjectFeature,
    BoundingBox,
    VisualEmbedding,
)
from app.models.enums import SceneType, ImportanceLevel
from app.core.analyzers.base import BaseParticipantAnalyzer
from app.core.analyzers.reference import BaselineReferenceAnalyzer
from app.core.analyzers.semantic_detector import SemanticDetector


class BaselineParticipantAnalyzer(BaseParticipantAnalyzer):
    """
    Standard computer vision analyzer for participant recreations.
    Extracts visual features, scene identity, extraneous elements, and CLIP embeddings.
    """

    def __init__(self):
        self._extractor = BaselineReferenceAnalyzer()

    def analyze(
        self,
        image_path: Path,
        participant_id: str,
        reference_profile: Optional[ReferenceProfile] = None,
    ) -> ParticipantProfile:
        with Image.open(image_path) as pil_img:
            pil_rgb = pil_img.convert("RGB")
            width, height = pil_rgb.size
            img_np = np.array(pil_rgb)

        # 1. Color Analysis
        color_feature = self._extractor._extract_color_profile(img_np)

        # 2. Composition and Spatial Analysis
        composition_feature = self._extractor._extract_composition_profile(img_np, width, height)

        # 3. Fine Detail Analysis
        detail_feature = self._extractor._extract_detail_profile(img_np)

        # 4. Lighting Feature
        lighting_feature = self._extractor._extract_lighting_profile(color_feature, composition_feature)

        # 5. Semantic Profile Analysis
        sem_detector = SemanticDetector.get_instance()
        sem_profile = sem_detector.analyze_semantic_profile(pil_rgb)

        # 6. Extract Salient Elements
        main_subjects, secondary_subjects, all_objects = self._extractor._identify_elements(
            img_np, width, height, composition_feature
        )

        # If semantic detector identified a clear subject domain, attach semantic tag
        detected_subject_type = sem_profile.get("main_subject", "general_subject")
        detected_subject_desc = sem_profile.get("main_subject_desc", "Visual subject")
        detected_scene_domain = sem_profile.get("scene_domain", "general_scene")
        extraneous_detected = sem_profile.get("extraneous_detected", [])

        if detected_subject_type == "sports_athlete":
            if main_subjects:
                main_subjects[0].name = "Football Player / Sports Athlete"
                main_subjects[0].category = "sports_athlete"
                main_subjects[0].notes = f"Sports player in athletic jersey. {main_subjects[0].notes or ''}"
            else:
                athlete_obj = ObjectFeature(
                    name="Football Player / Sports Athlete",
                    category="sports_athlete",
                    importance=ImportanceLevel.HIGH,
                    location="center",
                    relative_size=35.0,
                    confidence=0.95,
                    bounding_box=BoundingBox(xmin=0.20, ymin=0.10, xmax=0.80, ymax=0.90),
                    notes="Football player in athletic kit",
                )
                main_subjects = [athlete_obj]
                all_objects.insert(0, athlete_obj)
        elif detected_subject_type == "ceremonial_elephant":
            if main_subjects:
                main_subjects[0].name = "Decorated Ceremonial Elephant Recreation"
                main_subjects[0].category = "primary_subject"

        # Append explicit extraneous elements to participant's objects list for comparator inspection
        for ext_name in extraneous_detected:
            ext_obj = ObjectFeature(
                name=f"Extraneous: {ext_name}",
                category="extraneous_element",
                importance=ImportanceLevel.HIGH,
                location="center",
                relative_size=20.0,
                confidence=0.90,
                bounding_box=BoundingBox(xmin=0.15, ymin=0.15, xmax=0.85, ymax=0.85),
                notes=f"Detected extraneous visual element unrelated to reference: {ext_name}",
            )
            all_objects.append(ext_obj)

        # 7. Structure Feature
        structure_feature = StructureFeature(
            aspect_ratio=composition_feature.aspect_ratio,
            width=width,
            height=height,
            main_regions=[f"Zone-{k}" for k in composition_feature.quadrant_density.keys()],
            visual_complexity=round(
                float(min(1.0, (detail_feature.edge_density * 2.0 + detail_feature.texture_complexity) / 2.0)), 3
            ),
        )

        # 8. Scene Feature
        scene_type = SceneType.OUTDOOR if color_feature.average_brightness > 0.45 else SceneType.INDOOR
        if detected_scene_domain in ["kitchen_interior", "office_interior"]:
            scene_type = SceneType.INDOOR

        scene_feature = SceneFeature(
            scene_type=scene_type,
            location_type=sem_profile.get("environment_desc", "Participant recreation environment"),
            background=f"Background with {composition_feature.layout_description.lower()}",
            foreground=f"Salient subjects ({detected_subject_desc}) near ({composition_feature.focal_center[0]:.2f}, {composition_feature.focal_center[1]:.2f})",
            general_environment=sem_profile.get("environment_desc", f"Environment with {color_feature.lighting_description.lower() if color_feature.lighting_description else 'balanced lighting'}"),
            scene_context=f"{detected_subject_desc} in {sem_profile.get('environment_desc', 'general environment')}",
        )

        # 9. Visual Embedding (Normalized 512-dim CLIP descriptor or fallback)
        clip_emb = sem_profile.get("clip_embedding")
        if clip_emb:
            visual_embedding = VisualEmbedding(
                model_name="OpenAI-CLIP-ViT-B/32-Quantized",
                dimension=512,
                embedding=clip_emb,
                is_normalized=True,
            )
        else:
            visual_embedding = self._extractor._generate_visual_embedding(img_np)

        summary_scene = sem_profile.get("environment_desc", "unspecified scene")
        return ParticipantProfile(
            participant_id=participant_id,
            filename=image_path.name,
            dimensions=(width, height),
            main_subject=main_subjects[0] if main_subjects else None,
            secondary_subjects=secondary_subjects,
            objects=all_objects,
            scene=scene_feature,
            composition=composition_feature,
            colors=color_feature,
            lighting=lighting_feature,
            details=detail_feature,
            structure=structure_feature,
            visual_embedding=visual_embedding,
            detected_features=sem_profile.get("feature_matches", {}),
            semantic_summary=(
                f"Participant {participant_id} depicts {detected_subject_desc} in {summary_scene} with "
                f"{len(all_objects)} detected elements and {color_feature.dominant_colors[0].color_name if color_feature.dominant_colors else 'balanced'} dominant tone."
            ),
        )
