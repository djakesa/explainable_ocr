import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from skimage.segmentation import mark_boundaries


from torch.utils.data import DataLoader
from data.import_data import ImportData

from model.simple_cnn import SimpleCNN
from lime.super_pixels import superpixels



class regression():
    def __init__(self, batch_size, n_segments, compactness, target_class, alpha):
        self.batch_size = batch_size
        self.n_segments = n_segments
        self.compactness = compactness
        self.target_class = target_class
        self.alpha = alpha

    def _to_model_tensor(self, img_np):
        """Image (H,W,3) [0,1] -> Tensor (1,3,32,32) normalisé"""
        x = torch.tensor(img_np.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)
        x = F.interpolate(x, size=(32, 32), mode="bilinear", align_corners=False)
        x = (x - 0.5) / 0.5
        return x

    def bruit(self, image, seg, K, n_perturb=1000):
        """Masque aléatoirement certains superpixels en noir (0.0)"""
        images, Z = [], []
        for _ in range(n_perturb):
            z = np.ones(K, dtype=np.float32)
            nb_mask = np.random.randint(1, max(2, K//2))
            zeros = np.random.choice(K, size=nb_mask, replace=False)
            z[zeros] = 0
            Z.append(z)

            img_b = image.copy()
            for sp_id in zeros:
                mask = (seg == sp_id)
                img_b[mask] = 0.0   # <-- NOIR, et plus BLANC
            images.append(np.clip(img_b, 0, 1))
        return np.array(images), np.array(Z)


    def regression(self, image, seg):
        """Calcule les poids LIME des superpixels"""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SimpleCNN().to(device)
        model.load_state_dict(torch.load(
            "/home/jmerhrioui/soprasteria/poc/train/cnn_mnist_ckpt.pth",
            map_location=device))
        model.eval()

        K = int(seg.max() + 1)
        if self.target_class is None:
            x_orig = self._to_model_tensor(image).to(device)
            with torch.no_grad():
                logits = model(x_orig)
                self.target_class = torch.argmax(logits, dim=1).item()

        print(f"[DEBUG] Image: {image.shape}, superpixels={K}, classe={self.target_class}")

        # Générer perturbations
        images_bruitees, Z = self.bruit(image, seg, K, n_perturb=1000)

        # Scores sur la classe cible
        scores = []
        for img in images_bruitees:
            x = self._to_model_tensor(img).to(device)
            with torch.no_grad():
                logits = model(x)
                probs = F.softmax(logits, dim=1)[0]
            scores.append(probs[self.target_class].item())
        y = np.array(scores)

        # Poids gaussiens (kernel LIME)
        d = np.linalg.norm(Z - 1.0, axis=1)
        sigma = max(np.std(d), 1e-6)
        weights = np.exp(-(d**2) / (sigma**2))

        # Régression Lasso
        lasso = Lasso(alpha=self.alpha) # FAIBLE REGU donc plus de pixel diff 0 
        lasso.fit(Z, y, sample_weight=weights)
        w = lasso.coef_

        # Normaliser les poids dans [-1,1]
        if np.max(np.abs(w)) > 0:
            w = w / np.max(np.abs(w))

        print("[RESULTATS] Poids superpixels normalisés :", np.round(w, 3))
        return w


    def visualiser_top_superpixels(self, out_dir):

        os.makedirs(out_dir, exist_ok=True)
        if self.target_class is None:
            raise ValueError("Définis target_class dans regression(...).")

        # Charger le modèle
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SimpleCNN().to(device)
        model.load_state_dict(torch.load(
            "/home/jmerhrioui/soprasteria/poc/train/cnn_mnist__rota_ckpt.pth",
            #"/home/jmerhrioui/soprasteria/poc/train/cnn_mnist_ckpt.pth",
            map_location=device))
        model.eval()

        # Charger toutes les données test
        data = ImportData()
        #_, test_loader = data.load_data_mnist(batch_size=32)
        test_loader = data.load_data_rica(batch_size=32)
        #_, test_loader = data.load_data_mnist_rota(batch_size=32)

        correct_samples = []
        wrong_samples = []


        # --- Parcourir toutes les images test ---
        for x, y in test_loader:
            for i in range(len(x)):
                if y[i].item() != self.target_class:
                    continue
                img = x[i].numpy().transpose(1, 2, 0)  # (32,32,3)
                label = y[i].item()

                # prédiction
                x_in = x[i].unsqueeze(0).to(device)
                with torch.no_grad():
                    logits = model(x_in)
                    pred = torch.argmax(logits, dim=1).item()

                if pred == label:
                    correct_samples.append((img, label, pred))
                else:
                    wrong_samples.append((img, label, pred))

        print(f"[INFO] Trouvé {len(correct_samples)} corrects et {len(wrong_samples)} erreurs pour la classe {self.target_class}")

        # --- Sélection moitié/moitié ---
        n_total = min(self.batch_size, len(correct_samples) + len(wrong_samples))
        n_wrong = min(len(wrong_samples), n_total // 2)
        n_correct = n_total - n_wrong
        selected = wrong_samples[:n_wrong] + correct_samples[:n_correct]

        # --- Générer les superpixels ---
        from skimage.segmentation import slic
        from skimage.util import img_as_float

        for idx, (image, label, pred_class) in enumerate(selected):
            img = img_as_float(image)
            seg = slic(img, n_segments=self.n_segments, compactness=self.compactness,
                    sigma=0, start_label=0)

            # ---- Régression LIME ----
            w = self.regression(img, seg)

            # --- Normalisation image ---
            img_disp = (img - img.min()) / (img.max() - img.min() + 1e-8)

            # --- Normalisation poids ---
            w_abs = np.abs(w)
            w_norm = w_abs / (np.max(w_abs) + 1e-8)

            # --- Carte de chaleur RGB ---
            heatmap = np.zeros((*seg.shape, 3), dtype=np.float32)
            cmap = plt.cm.get_cmap('hot')
            for sp_id in range(int(seg.max() + 1)):
                heatmap[seg == sp_id] = cmap(w_norm[sp_id])[:3]

            # --- Superpixels actifs ---
            partial = np.ones_like(img_disp) * 0.5
            mask_nonzero = np.isin(seg, np.where(w != 0)[0])
            partial[mask_nonzero] = img_disp[mask_nonzero]

            # --- Contours ---
            img_contours = mark_boundaries(img_disp, seg, color=(1, 0, 0), mode='subpixel')
            heatmap_contours = mark_boundaries(heatmap, seg, color=(1, 0, 0), mode='subpixel')
            partial_contours = mark_boundaries(partial, seg, color=(1, 0, 0), mode='subpixel')

            # --- Affichage ---
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            color_title = "green" if pred_class == label else "red"
            fig.suptitle(f"Vérité : {label} — Prédiction : {pred_class}",
                        fontsize=14, y=1.05, color=color_title)

            axes[0].imshow(img_contours); axes[0].axis("off"); axes[0].set_title("Image + contours")
            axes[1].imshow(heatmap_contours); axes[1].axis("off"); axes[1].set_title("Importance superpixels")
            sm = plt.cm.ScalarMappable(cmap='hot', norm=plt.Normalize(vmin=0, vmax=1))
            cbar = plt.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.04); cbar.set_label("Poids normalisés")
            axes[2].imshow(partial_contours); axes[2].axis("off"); axes[2].set_title("Superpixels actifs")

            #out_path = os.path.join(out_dir, f"lime_{idx}_true{label}_pred{pred_class}.png")
            out_path = os.path.join(out_dir, f"lime_{idx}_true{label}.png")
            plt.savefig(out_path, dpi=300, bbox_inches="tight"); plt.close(fig)
            print(f"[INFO] Image sauvegardée : {out_path}")


    def score_recouvrement(self, image, seg, w):
        # déterminer quels superpixels appartiennent au chiffre (zone blanche)
        gray = image.mean(axis=2)
        mask_digit = gray > 0.1

        digit_sp = []
        for sp_id in range(int(seg.max() + 1)):
            if (mask_digit & (seg == sp_id)).sum() > 0:
                digit_sp.append(sp_id)
        digit_set = set(digit_sp)

        # superpixels triés par importance absolue
        # top = superpixels prédits importants par LIME
        top_sp = np.argsort(np.abs(w))[::-1][:len(digit_set)]
        top_set = set(top_sp.tolist())

        # intersection
        inter = len(top_set & digit_set)

        # score = proportion de superpixels du chiffre correctement captés
        score = inter / len(digit_set) if len(digit_set) > 0 else 0.0

        print(f"[SCORE] Digit={len(digit_set)}, Top={len(top_set)}, "
            f"Corrects={inter}, Score={score:.2f}")

        return score




