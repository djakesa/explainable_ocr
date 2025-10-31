import numpy as np
from PIL import Image
import sys, os, gc
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from superpixels.sp import superpixels
from lime.regression import regression
from model.model import model
from tqdm import tqdm

# --- PARAMÈTRES GLOBAUX ---
l_min, l_max = 6, 7
c_min, c_max = 16, 22
diff = (c_max - c_min)
L = []

# Calcul des indices à tester
for e in range(l_min, l_max + 1):
    numero = np.linspace(e * 48 + c_min, e * 48 + c_max, diff + 1, dtype=int).tolist()
    L.extend(numero)

numero = L
print("Patchs à tester :", numero)

path = "/home/jmerhrioui/soprasteria/semaine_2/data/dataset_y/images/validation/validation_image_4.png"
patch_size = 40
sp_batch_size = 3  # pour superpixels/regression (garde)
n_segments = 50
compactness = 5
n_perturb = 10     #  nombre d'images complètes que tu veux produire

# --- INITIALISATION DES OBJETS ---
sp = superpixels(batch_size=sp_batch_size, n_segments=n_segments, compactness=compactness)
reg = regression(batch_size=sp_batch_size, n_segments=n_segments, compactness=compactness, alpha=1)
model_instance = model(num=0)  # instance YOLO (chargée une fois)
accs = []

# --- CHARGEMENT IMAGE + PATCHS ---
liste, imgs, img_base_striee = sp.load_image(patch_size)
Image.fromarray((img_base_striee * 255).astype(np.uint8)).save(
    "/home/jmerhrioui/soprasteria/semaine_2/test/image_base_striee.png"
)
visus, segs, borders = sp.superpixels()

base = Image.open(path).convert("RGB")

# --- TAILLE ET DÉCOUPAGE ---
if imgs.ndim == 4:
    H, W = imgs.shape[1:3]
else:
    H, W = imgs.shape[0:2]
print(f"Image complète taille : {H}x{W}")
cols = W // patch_size

# --- PRÉPARATION : n_perturb images complètes à construire ---
# Au lieu de générer 1 patch à la fois -> on construit n_perturb images finales,

final_images = [base.copy() for _ in range(n_perturb)]

# --- BOUCLE SUR LES PATCHS, puis collage sur chaque image k ---
for num in tqdm(numero, desc="Bruitage et collage par patch"):
    if num >= len(liste):
        print(f" Patch {num} hors limites ({len(liste)} total).")
        continue

    image_patch = liste[num]  # (1,40,40,3)
    seg_patch   = segs[num]   # (40,40)

    #  Ici on demande n_perturb perturbations pour CE patch
    images_noise, Z = reg.bruit(image_patch, seg_patch, K=n_segments, n_perturb=n_perturb)

    # Coordonnées du patch dans l’image complète
    row, col = num // cols, num % cols
    x, y = col * patch_size, row * patch_size

    # Pour chaque image finale k, coller la perturbation k de ce patch
    for k in range(n_perturb):
        patch_np = (np.clip(images_noise[k], 0, 1) * 255).round().astype(np.uint8)
        if patch_np.ndim == 4 and patch_np.shape[0] == 1:
            patch_np = patch_np[0]  # (40,40,3)

        patch_img = Image.fromarray(patch_np)
        if patch_img.size != (patch_size, patch_size):
            patch_img = patch_img.resize((patch_size, patch_size), Image.NEAREST)

        final_images[k].paste(patch_img, (x, y))

    # petit ménage
    del images_noise, Z
    gc.collect()

# --- INFÉRENCE YOLO EN MICRO-BATCHS SUR LES n_perturb IMAGES ---
inference_batch_size = 16  # ajuste si besoin (8–32)

# --- VISUALISATION DES IMAGES BRUITÉES ---
output_dir = "/home/jmerhrioui/soprasteria/semaine_2/test/images_bruitees"
os.makedirs(output_dir, exist_ok=True)

print(f"\n Sauvegarde de {len(final_images)} images bruitées dans {output_dir}")

for i, img in enumerate(final_images):
    img_path = os.path.join(output_dir, f"image_bruitee_{i+1}.png")
    img.save(img_path)
    print(f"  → Image bruitée {i+1} enregistrée : {img_path}")

accs = model_instance.test_in_batches(final_images, batch_size=inference_batch_size)

print(f" Fin du test — {len(accs)} scores calculés (={n_perturb} images complètes)")
