"""
trocr_model.py
==============

Implémentation TrOCR conforme à l'interface OCRModel.
- Chargement depuis un dossier local ou HF
- Prédiction image -> texte
- Tokens + probabilités (token-level)
"""

from typing import Dict, List
import torch
from PIL import Image

from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from ocr.base import OCRModel


class TrOCRModel(OCRModel):
    def __init__(
        self,
        model_path: str = "model/trocr-base-handwritten",
        device: str | None = None,
    ):
        """
        Parameters
        ----------
        model_path : str
            Chemin local vers le modèle TrOCR ou nom HF.
        device : str | None
            "cuda", "cpu" ou None (auto)
        """

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Processor = image processor + tokenizer
        self.processor = TrOCRProcessor.from_pretrained(model_path)

        # Modèle encodeur-décodeur
        self.model = VisionEncoderDecoderModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

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
                "token_probs": List[float]
            }
        """

        if not isinstance(image, Image.Image):
            raise TypeError("predict() attend une image PIL.Image")

        # --- Prétraitement image ---
        pixel_values = self.processor(
            images=image,
            return_tensors="pt"
        ).pixel_values.to(self.device)

        # --- Génération autoregressive ---
        outputs = self.model.generate(
            pixel_values,
            return_dict_in_generate=True,
            output_scores=True,
            max_length=64,
        )

        # Séquence complète (inclut <bos> et <eos>)
        generated_ids = outputs.sequences[0]

        # Texte final
        text = self.processor.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True
        )

        # --- Token-level probabilities ---
        tokens: List[str] = []
        token_probs: List[float] = []

        # outputs.scores[t] = logits pour le token généré à l'étape t+1
        # (car t=0 correspond au premier token après <bos>)
        for t, scores_t in enumerate(outputs.scores):
            probs_t = torch.softmax(scores_t[0], dim=-1)

            token_id = generated_ids[t + 1].item()
            prob = probs_t[token_id].item()

            token_str = self.processor.tokenizer.decode(
                token_id,
                skip_special_tokens=True
            )

            # On ignore les tokens vides
            if token_str.strip():
                tokens.append(token_str)
                token_probs.append(prob)

        return {
            "text": text,
            "tokens": tokens,
            "token_probs": token_probs,
        }


# ---------------------------------------------------------------------
# Test local (sans interface)
# ---------------------------------------------------------------------

if __name__ == "__main__":
    """
    Lancer depuis la racine du projet :

    python -m ocr.trocr_model
    """

    from pathlib import Path

    img_path = Path("examples/test_anglais.png")
    if not img_path.exists():
        raise FileNotFoundError(f"Image de test introuvable : {img_path}")

    image = Image.open(img_path).convert("RGB")

    ocr = TrOCRModel()
    result = ocr.predict(image)

    print("\n--- OCR RESULT ---")
    print("Texte reconnu :", result["text"])
    print("\nTokens :")
    for t, p in zip(result["tokens"], result["token_probs"]):
        print(f"  {t!r} -> {p:.4f}")
