import os
import numpy as np
from PIL import Image
from skimage.segmentation import slic, mark_boundaries
from skimage.util import img_as_float


class superpixels:
    def __init__(self, batch_size, n_segments, compactness):
        self.batch_size = batch_size
        self.n_segments = n_segments
        self.compactness = compactness

    def load_image(self, size):
        path = "/home/jmerhrioui/soprasteria/bee/semaine_2/data/dataset_y/images/validation/validation_image_4.png"
        img = Image.open(path).convert("RGB")
        img = img_as_float(np.array(img))
        H, W, _ = img.shape

        n = H // size
        m = W // size
        liste = []

        for i in range(n):
            for j in range(m):
                img_crop = img[i*size:(i+1)*size, j*size:(j+1)*size, :]
                liste.append(img_crop)

        seg = np.zeros((H, W), dtype=int)
        sp_id = 0
        for i in range(0, H, size):
            for j in range(0, W, size):
                seg[i:i+size, j:j+size] = sp_id
                sp_id += 1

        img_base_striee = mark_boundaries(img, seg, color=(1, 0, 0), mode='subpixel')
        return liste, img, img_base_striee

    def superpixels(self):
        nbre_solo = 1
        liste, _, _ = self.load_image(40)
        visus, segs, borders = [], [], []

        for img in liste:
            if np.allclose(img, img[0, 0, :]):
                seg = slic(img, n_segments=nbre_solo, compactness=self.compactness, start_label=0)
            else:
                seg = slic(
                    img,
                    n_segments=min(self.n_segments, img.shape[0]*img.shape[1]),
                    compactness=self.compactness,
                    start_label=0
                )
            vis = mark_boundaries(img, seg, color=(1, 0, 0))
            visus.append(img)
            segs.append(seg)
            borders.append(vis)

        return visus, segs, borders

    def save_superpixels(self, out_path):
        visus, _, borders = self.superpixels()
        _, img, _ = self.load_image(40)

        h, w, _ = img.shape
        grid_size = 40
        rows = h // grid_size
        cols = w // grid_size

        output_img = Image.new("RGB", (w, h))
        idx = 0
        for i in range(rows):
            for j in range(cols):
                if idx >= len(borders):
                    break
                patch = (borders[idx] * 255).astype(np.uint8)
                patch_img = Image.fromarray(patch)
                output_img.paste(patch_img, (j * grid_size, i * grid_size))
                idx += 1

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        output_img.save(out_path)
        print(f"Saved superpixels image to {out_path}")




