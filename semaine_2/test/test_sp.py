import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from superpixels.sp import superpixels


sp = superpixels(batch_size=3, n_segments=50, compactness=5)
# batch_size est ici le nombre d'images à récupérer dans une categorie
#visus, segs , vis  = sp.superpixels()

# Afficher et sauvegarder les superpixels
sp.save_superpixels()

