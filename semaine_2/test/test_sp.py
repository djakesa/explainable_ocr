import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
import os
import sys
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from superpixels.sp import superpixels

sp = superpixels(batch_size=3, n_segments=50, compactness=5) # batch size ne sert plus à rien ici


def test_creation_pachs(sp=sp):
    # liste contient les patches découpés
    # imgs est l'image d'origine
    size=40 # taille des patches : size * size
    liste, imgs, img_base_striee = sp.load_image(size)
    assert len(liste) == (imgs.shape[0]//size) * (imgs.shape[1]//size)

def test_generation_sp(sp=sp):
    out_path="/home/jmerhrioui/soprasteria/bee/semaine_2/test/superpixels_output.png"
    # Afficher et sauvegarder les superpixels
    sp.save_superpixels(out_path)
    assert os.path.exists(out_path)

    

