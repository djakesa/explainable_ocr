# ocr/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any
from PIL import Image


@dataclass
class OCRPrediction:
    text: str
    tokens: List[str]
    token_probs: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "tokens": self.tokens,
            "token_probs": self.token_probs,
        }


class OCRBase(ABC):
    """
    Contrat commun pour tous les moteurs OCR utilisés dans l'app.
    LIME a besoin de token_probs pour un token_index donné.
    """

    @abstractmethod
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Doit retourner un dict:
        {
            "text": str,
            "tokens": List[str],
            "token_probs": List[float]
        }
        """
        raise NotImplementedError

    def predict_token_prob(self, image: Image.Image, token_index: int) -> float:
        """
        Helper standard (utilisé par LIME) : renvoie P(token_index).
        Par défaut, appelle predict(). Les modèles peuvent override
        si une version plus rapide est possible.
        """
        out = self.predict(image)
        probs = out.get("token_probs", [])
        if token_index < 0 or token_index >= len(probs):
            return 0.0
        return float(probs[token_index])
