import os
import sys
import numpy as np
from PIL import Image
import pytest

# Import des modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from superpixels.sp import superpixels
from lime.regression import regression


@pytest.mark.parametrize("patch_size,n_segments,compactness", [(40, 50, 5)]) # tester différentes configurations
def test_image_bruitee_complete(tmp_path, patch_size, n_segments, compactness):
    path = "/home/jmerhrioui/soprasteria/bee/semaine_2/data/dataset_y/images/validation/validation_image_4.png"
    out_path = tmp_path / "image_bruitee_complete.png"

    batch_size = 3
    sp = superpixels(batch_size=batch_size, n_segments=n_segments, compactness=compactness)
    reg = regression(batch_size=batch_size, n_segments=n_segments, compactness=compactness, alpha=1)

    # Chargement des images
    liste, imgs, _ = sp.load_image(patch_size)
    visus, segs, _ = sp.superpixels()

    # Génération des indices de test
    l_min, l_max = 6, 7
    c_min, c_max = 16, 22
    L = []
    for e in range(l_min, l_max + 1):
        diff = c_max - c_min
        L.extend(np.linspace(e * 48 + c_min, e * 48 + c_max, diff + 1, dtype=int).tolist())

    # Image de base
    base = Image.open(path).convert("RGB")
    modifiee = base.copy()
    H, W = imgs.shape[1:3]
    cols = max(1, W // patch_size)


    for num in L:
        image_patch = liste[num]
        seg_patch = segs[num]

        images_noise, _ = reg.bruit(image_patch, seg_patch, K=n_segments, n_perturb=10)
        patch_np = (np.clip(images_noise[0], 0, 1) * 255).astype(np.uint8)
        if patch_np.ndim == 4:
            patch_np = patch_np[0]
        patch_img = Image.fromarray(patch_np).resize((patch_size, patch_size), Image.NEAREST)

        row, col = divmod(num, cols)
        x, y = col * patch_size, row * patch_size
        modifiee.paste(patch_img, (x, y))

    modifiee.save(out_path)
    assert os.path.exists(out_path), f"L'image bruitée n'a pas été générée : {out_path}"
