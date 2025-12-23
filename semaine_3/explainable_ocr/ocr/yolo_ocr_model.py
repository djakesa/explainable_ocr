"""
yolo_ocr_model.py
=================

OCR basé sur YOLO (détection de caractères).
NON séquentiel → pas compatible LIME token-level.
"""

from typing import Dict, List
import torch
import numpy as np
from PIL import Image

from ocr.base import OCRBase


class YOLOOCRModel(OCRBase):
    """
    OCR par détection basé sur YOLO.
    """

    ocr_type = "detection"

    def __init__(
        self,
        model_path: str,
        device: str | None = None,
        conf_threshold: float = 0.25,
        img_size: int = 1280,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.conf_threshold = conf_threshold
        self.img_size = img_size

        print("🔍 Chargement du modèle YOLO OCR…")

        # Chargement YOLOv5 via torch.hub
        self.model = torch.hub.load(
            "ultralytics/yolov5",
            "custom",
            path=model_path,
            force_reload=False,
            trust_repo=True,
        )

        self.model.to(self.device)
        self.model.eval()

        self.class_names = self.model.names
        print("Classes YOLO :", self.class_names)

    @torch.no_grad()
    def predict(self, image: Image.Image) -> Dict[str, List]:
        """
        OCR sur une image PIL.

        Returns
        -------
        dict :
            {
                "text": str,
                "tokens": List[str],
                "token_probs": List[float],
                "boxes": List[List[float]]
            }
        """

        if not isinstance(image, Image.Image):
            raise TypeError("predict() attend une image PIL.Image")

        print("🧠 OCR en cours…")

        img = np.array(image)

        # Inference YOLO
        results = self.model(img, size=self.img_size)

        if len(results.xyxy) == 0 or len(results.xyxy[0]) == 0:
            return {
                "text": "",
                "tokens": [],
                "token_probs": [],
                "boxes": [],
            }

        detections = results.xyxy[0].cpu().numpy()
        print("Detections brutes :", detections)

        boxes = []
        tokens = []
        scores = []

        for x1, y1, x2, y2, conf, cls_id in detections:
            if conf < self.conf_threshold:
                continue

            char = self.class_names.get(int(cls_id), "")
            if not char:
                continue

            boxes.append([float(x1), float(y1), float(x2), float(y2)])
            tokens.append(char)
            scores.append(float(conf))

        # Reconstruction naïve du texte (tri gauche → droite)
        if boxes:
            order = np.argsort([b[0] for b in boxes])
            tokens = [tokens[i] for i in order]
            scores = [scores[i] for i in order]
            boxes = [boxes[i] for i in order]

        text = "".join(tokens)

        return {
            "text": text,
            "tokens": tokens,
            "token_probs": scores,  # utilisé comme "confidence"
            "boxes": boxes,
        }


# ---------------------------------------------------------------------
# Test local
# ---------------------------------------------------------------------

if __name__ == "__main__":
    from pathlib import Path

    img_path = Path("examples/test_anglais.png")
    model_path = Path("model/jhmi_yolov5_OCR_model.pt")

    if not img_path.exists():
        raise FileNotFoundError(img_path)
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    image = Image.open(img_path).convert("RGB")

    ocr = YOLOOCRModel(model_path=str(model_path))
    result = ocr.predict(image)

    print("\n--- YOLO OCR RESULT ---")
    print("Texte reconnu :", result["text"])
    print("\nTokens :")
    for t, p in zip(result["tokens"], result["token_probs"]):
        print(f"  {t!r} -> {p:.4f}")
