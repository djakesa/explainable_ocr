import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from lime.super_pixels import superpixels
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.import_data import ImportData


ImportData = ImportData()
# trainloader, test_loader = ImportData.load_data_mnist(batch_size=32) # on charge déjà image dans superpixels.load_image()
# test_loader = ImportData.load_data_rica(batch_size=32)

from lime.super_pixels import superpixels

sp = superpixels(batch_size=3, n_segments=50, compactness=15, target_class=7, type_data="image_seule")
# batch_size est ici le nombre d'images à récupérer dans une categorie
visus, segs , vis  = sp.superpixels()

# Afficher et sauvegarder les superpixels
sp.save_superpixels()

