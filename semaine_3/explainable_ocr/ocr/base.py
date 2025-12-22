"""
base.py
=======

Interface abstraite pour tous les modèles OCR du projet.

Objectif :
- Garantir une sortie standardisée
- Permettre le multi-modèle (TrOCR, PARSeq, CRNN, etc.)
- Isoler l'explicabilité (LIME) de la logique OCR
"""

from typing import Dict, List
from PIL import Image


class OCRModel:
    """
    Interface commune à TOUS les modèles OCR.
    """

    def predict(self, image: Image.Image) -> Dict[str, List]:
        """
        Effectue une prédiction OCR sur une image.

        Parameters
        ----------
        image : PIL.Image.Image

        Returns
        -------
        dict :
            {
                "text": str,
                "tokens": List[str],
                "token_probs": List[float]
            }
        """
        raise NotImplementedError(
            "Chaque modèle OCR doit implémenter la méthode predict()"
        )
