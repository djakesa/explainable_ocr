import numpy as np
from PIL import Image
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from superpixels.sp import superpixels
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lime.regression import regression

l_min = 6
l_max = 7
c_min = 16
c_max = 22
diff = (c_max - c_min)
L= []
# e correspond à la ligne
for e in range (l_min, l_max+1):
    print(e)
    numero = np.linspace(e*48+c_min,e*48+c_max,diff+1,dtype = int).tolist()
    print("test", numero)
    L.extend(numero)
print (L)
numero = L 
path = "/home/jmerhrioui/soprasteria/semaine_2/data/dataset_y/images/validation/validation_image_4.png"
patch_size = 40
batch_size=3
n_segments=50
compactness=5

sp  = superpixels(batch_size=batch_size, n_segments=n_segments, compactness=compactness)
reg = regression(batch_size=batch_size, n_segments=n_segments, compactness=compactness, alpha=1)

# Charge: liste des patches (float [0,1]), image complète (1,H,W,3), image + contours
liste, imgs, img_base_striee = sp.load_image(patch_size)

# Sauvegarde de la grille (debug)
Image.fromarray((img_base_striee*255).astype(np.uint8)).save(
    "/home/jmerhrioui/soprasteria/semaine_2/test/image_base_striee.png"
)

# Segmentation par patch
visus, segs, borders = sp.superpixels()

# Image de fond à modifier 
base = Image.open(path).convert("RGB")
modifiee = base.copy()

H, W = imgs.shape[0], imgs.shape[1]
print("Image complète taille (H,W):", H, W)
cols = W // patch_size

for num in numero:
    # patch d’entrée + segmentation du patch
    image_patch = liste[num]        # (1,40,40,3) float [0,1]
    seg_patch   = segs[num]         # (40,40)

    # LIME:  images bruitées (float [0,1])
    images_noise, Z = reg.bruit(image_patch, seg_patch, K=n_segments, n_perturb=1000)

    # Prendre la 2ère image bruitée parmis les n_perturb et la convertir pour PIL
    patch_np = (np.clip(images_noise[1], 0, 1) * 255).round().astype(np.uint8)
    if patch_np.ndim == 4 and patch_np.shape[0] == 1:
        patch_np = patch_np[0]  # (40,40,3)
    patch_img = Image.fromarray(patch_np)

    # Coordonnées (x,y) du patch dans l’image complète
    row = num // cols
    col = num % cols
    x = col * patch_size
    y = row * patch_size

    # (remplacement direct)
    if patch_img.size != (patch_size, patch_size):
        patch_img = patch_img.resize((patch_size, patch_size), Image.NEAREST)
    modifiee.paste(patch_img, (x, y))

out_path = "/home/jmerhrioui/soprasteria/semaine_2/test/image_bruitee_complete.png"
modifiee.save(out_path)
print(f" Image bruitée sauvegardée -> {out_path}")
