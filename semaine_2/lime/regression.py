import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler


class regression:
    def __init__(self, batch_size, n_segments, compactness, alpha):
        self.batch_size = batch_size
        self.n_segments = n_segments
        self.compactness = compactness
        self.alpha = alpha

    def _to_model_tensor(self, img_np):
        x = torch.tensor(img_np.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0)
        x = F.interpolate(x, size=(32, 32), mode="bilinear", align_corners=False)
        x = (x - 0.5) / 0.5
        return x

    def bruit(self, image, seg, K, n_perturb=1000):
        images, Z = [], []
        if image.ndim == 4:
            image = image.squeeze(0)
        baseline = image.mean(axis=(0, 1))

        for _ in range(n_perturb):
            z = np.ones(K, dtype=np.float32)
            zeros = np.random.choice(K, size=np.random.randint(1, max(2, K // 2)), replace=False)
            z[zeros] = 0
            Z.append(z)

            img_b = image.copy()
            for sp_id in zeros:
                img_b[seg == sp_id] = baseline
            images.append(np.clip(img_b, 0, 1))

        return np.array(images), np.array(Z)

    def regression_jacques(self, accs_all, Z_fused):
        n_images = len(accs_all)
        assert n_images == len(Z_fused), f"accs_all ({n_images}) ≠ Z_fused ({len(Z_fused)})"

        Z_concat = np.vstack(Z_fused)
        y = np.asarray(accs_all, dtype=float).ravel()

        Z_concat = np.clip(np.nan_to_num(Z_concat), 0.0, 1.0)
        y = np.nan_to_num(y)

        d = np.linalg.norm(Z_concat - 1.0, axis=1)
        sigma = max(np.std(d), 1e-6)
        weights = np.exp(-(d ** 2) / (sigma ** 2))
        weights[weights < 1e-8] = 1e-8

        Z_scaled = StandardScaler().fit_transform(Z_concat)
        lasso = Lasso(alpha=self.alpha, max_iter=10000)
        lasso.fit(Z_scaled, y, sample_weight=weights)
        w = lasso.coef_

        if np.max(np.abs(w)) > 0:
            w /= np.max(np.abs(w))

        print(f"[INFO] Régression LIME : {n_images} images, {len(w)} superpixels")
        return w
