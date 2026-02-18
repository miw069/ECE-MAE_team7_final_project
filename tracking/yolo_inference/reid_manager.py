"""Re-identification manager using a lightweight timm feature extractor.

Extracts appearance embeddings from person crops and matches them
against a reference embedding via cosine similarity.
"""

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

import config


class ReIDManager:
    """Extracts and compares appearance embeddings for person Re-ID."""

    def __init__(
        self,
        model_name: str = config.REID_MODEL,
        device: Optional[str] = None,
        threshold: float = config.REID_SIMILARITY_THRESHOLD,
    ):
        import timm

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.threshold = threshold

        print(f"[ReID] Loading {model_name} on {device} ...")
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        )
        self.model.eval()
        self.model.to(self.device)

        # Get expected input size from model config
        data_cfg = timm.data.resolve_model_data_config(self.model)
        self._transform = timm.data.create_transform(**data_cfg, is_training=False)

        print(f"[ReID] Model ready.")

    def extract(self, frame: np.ndarray, bbox: tuple) -> Optional[np.ndarray]:
        """Extract an L2-normalised embedding from a person crop.

        Args:
            frame: Full BGR frame.
            bbox:  (x1, y1, x2, y2) bounding box in pixels.

        Returns:
            1-D numpy float32 embedding, or None if the crop is too small.
        """
        x1, y1, x2, y2 = bbox
        # Clamp to frame bounds
        h, w = frame.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        crop_w, crop_h = x2 - x1, y2 - y1
        if crop_w < 32 or crop_h < 32:
            return None

        crop_bgr = frame[y1:y2, x1:x2]

        # Convert BGR → RGB PIL image for timm transform
        from PIL import Image
        crop_rgb = crop_bgr[:, :, ::-1].copy()
        pil_img = Image.fromarray(crop_rgb)

        tensor = self._transform(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            feat = self.model(tensor)  # (1, D)

        feat = F.normalize(feat, dim=1)
        return feat.squeeze(0).cpu().numpy().astype(np.float32)

    def find_best_match(
        self,
        reference: np.ndarray,
        candidates: list,
    ) -> Optional[int]:
        """Find the candidate whose embedding best matches *reference*.

        Args:
            reference:  L2-normalised reference embedding (1-D np.ndarray).
            candidates: List of (track_id, embedding) pairs where embedding
                        may be None (skipped).

        Returns:
            track_id of the best match if its cosine similarity >= threshold,
            else None.
        """
        best_id: Optional[int] = None
        best_sim: float = self.threshold - 1e-9  # must exceed threshold

        ref = torch.from_numpy(reference)

        for track_id, emb in candidates:
            if emb is None:
                continue
            cand = torch.from_numpy(emb)
            sim = float(F.cosine_similarity(ref.unsqueeze(0), cand.unsqueeze(0)))
            if sim > best_sim:
                best_sim = sim
                best_id = track_id

        return best_id
