import numpy as np
from skimage.segmentation import slic, mark_boundaries
from skimage.util import img_as_float
from skimage.transform import resize
import torch
import os, sys
from matplotlib import pyplot as plt
from PIL import Image
import random

# Pour importer classe ImportData
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.import_data import ImportData


class superpixels:
    def __init__(self, batch_size, n_segments, compactness, target_class,type_data):
        """
        batch_size: nombre d'images à récupérer
        n_segments: nombre de superpixels SLIC
        compactness: compacité SLIC
        target_class: si spécifié, ne prend que des images de cette classe MNIST (0-9)
        """
        self.batch_size = batch_size
        self.n_segments = n_segments
        self.compactness = compactness
        self.target_class = target_class
        self.type_data=type_data

    def load_image(self):
        """Charge batch_size images MNIST (32x32, 3 canaux)"""
        data = ImportData()
        if self.type_data=="mnist":
           _, test_loader = data.load_data_mnist(batch_size=128)
        elif self.type_data=="mnist_rota":
           _, test_loader = data.load_data_mnist_rota(batch_size=128)
        elif self.type_data=="rica":
            test_loader = data.load_data_rica(batch_size=32)
        elif self.type_data == "image_seule":
            image_path = "/home/jmerhrioui/soprasteria/poc/data/dataset_y/images/train/train_image_33.png"
            img = Image.open(image_path).convert("RGB")
            img = img.resize((32, 32))  # même format que MNIST/RICA
            img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            return img_tensor.numpy().transpose(0, 2, 3, 1)  # [B, 32, 32, 3]

        else:
            raise ValueError(f"type_data non reconnu : {self.type_data}")

        all_images = []
        all_labels = []
        for x, y in test_loader:
            all_images.append(x)
            all_labels.append(y)
        all_images = torch.cat(all_images)  # [N, 3, 32, 32]
        all_labels = torch.cat(all_labels).numpy()

        # Filtrage par classe si demandé
        if self.target_class is not None:
            idx = np.where(all_labels == self.target_class)[0]
        else:
            idx = np.arange(len(all_labels))

        if len(idx) < self.batch_size:
            raise ValueError(f"Pas assez d'images dans la classe {self.target_class}")

        chosen = np.random.choice(idx, size=self.batch_size, replace=False)
        imgs = all_images[chosen].numpy()  # [B, 3, 32, 32]

        # Passer en [B, 32, 32, 3] pour affichage
        imgs = imgs.transpose(0, 2, 3, 1)
        return imgs

    def save_image(self, out_path="/home/jmerhrioui/soprasteria/poc/test/test_img/mnist_imgs.png"):
        """Affiche les images brutes"""
        imgs = self.load_image()
        fig, axes = plt.subplots(1, len(imgs), figsize=(2 * len(imgs), 2))
        if len(imgs) == 1:
            axes = [axes]
        for ax, img in zip(axes, imgs):
            ax.imshow((img * 0.5 + 0.5).clip(0, 1))  # dénormalisation légère
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Images MNIST sauvegardées dans {out_path}")

    def superpixels(self):
        x = self.load_image()
        visus = []
        segs = []
        borders = []
        if self.type_data != "image_seule":
            for i in range(self.batch_size):
                img = x[i]
                img = img_as_float(img)

                seg = slic(
                    img,
                    n_segments=self.n_segments,
                    compactness=self.compactness,
                    sigma=0,
                    start_label=0
                )

                vis = mark_boundaries(img, seg, color=(1, 0, 0), mode='subpixel')

                visus.append(img)
                segs.append(seg)
                borders.append(vis)
        else:
            img = x[0]
            img = img_as_float(img) 
            seg = slic(
                img,
                n_segments=self.n_segments,
                compactness=self.compactness,
                sigma=0,
                start_label=0
            )
            vis = mark_boundaries(img, seg, color=(1, 0, 0), mode='subpixel')
            visus.append(img)
            segs.append(seg)
            borders.append(vis)

        return visus, segs, borders




    def save_superpixels(self, out_path="/home/jmerhrioui/soprasteria/poc/test/test_img/superpixels.png"):
        visus, _, borders = self.superpixels()
        if self.type_data != "image_seule":
            fig, axes = plt.subplots(1, len(visus), figsize=(2 * len(visus), 2))
            if len(visus) == 1:
                axes = [axes]
            for ax, vis, bord in zip(axes, visus, borders):
                ax.imshow(bord)  # <-- afficher directement l'image avec les bordures
                ax.axis('off')
        else:
            # Cas image unique
            fig, ax = plt.subplots(1, 1, figsize=(4, 4))
            print( borders)
            ax.imshow(borders[0])  # Utiliser la première image uniquement
            ax.axis('off')
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Superpixels sauvegardés dans {out_path}")

