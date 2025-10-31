import torch
from ultralytics import YOLO
import numpy as np
from PIL import Image
import os , sys

class model_jacques:
    def __init__(self, name):
        self.name = name

        #  Chargement unique du modèle YOLO
        PATH = "/home/jmerhrioui/soprasteria/semaine_2/model/best.pt"
        self.model = torch.hub.load(
            '/home/jmerhrioui/soprasteria/yolov5',
            'custom',
            path=PATH,
            source='local'
        ).to('cpu')
        self.model.conf = 0.2  # seuil de confiance
        self.model.iou = 0.4    # seuil d’IoU NMS

        torch.set_num_threads(os.cpu_count())
        torch.backends.mkldnn.enabled = True

    def preprocess(self, img):
        """Convertit les images PIL ou numpy pour YOLO."""
        if isinstance(img, np.ndarray):
            return img
        elif isinstance(img, Image.Image):
            return np.array(img)
        else:
            raise TypeError(f"Type d'image non supporté : {type(img)}")

    def test(self, images):
        """
        Accepte une image unique ou une liste d’images.
        Retourne une liste de scores (même si une seule image).
        """
        #  Uniformisation du type d’entrée
        if not isinstance(images, list):
            images = [images]

        imgs_ready = [self.preprocess(img) for img in images]

        #  Inférence en batch
        results = self.model(imgs_ready)

        scores = []
        # YOLOv5 : results est un seul objet Detections
        for i in range(len(results.pandas().xyxy)):
            df = results.pandas().xyxy[i]
            # Filtrer le dataframe sur la colonne "selection"
            filtered_df = df[df["name"] == self.name]

            if not filtered_df.empty:
                # On prend la première correspondance (ou plusieurs selon ton cas)
                confidence_value = float(filtered_df.iloc[0]["confidence"])
                scores.append(confidence_value)
            else:
                scores.append(0.0)  # Pas de détection pour cette image

        return scores

    def test_in_batches(self, images, batch_size):
        """
        Infère sur de grands ensembles d’images en micro-batchs
        pour éviter les dépassements mémoire.
        """
        scores = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size]
            batch_scores = self.test(batch)
            scores.extend(batch_scores)
            torch.cuda.empty_cache()  # même sur CPU, libère la mémoire PyTorch
        return scores
