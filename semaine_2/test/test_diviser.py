import numpy as np
from skimage.segmentation import slic, mark_boundaries
from skimage.util import img_as_float
from skimage.transform import resize
import torch
import os, sys
from matplotlib import pyplot as plt
from PIL import Image

def load_image(path,size):
    #charger une seule image 
    liste = []
    img = Image.open(path).convert("RGB")
    img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    imgs = img_tensor.numpy().transpose(0, 2, 3, 1)
    print(imgs.shape[1]//size, imgs.shape[2]//size)
    n = imgs.shape[1]//size
    m = imgs.shape[2]//size
    for i in range (n):
        for j in range (m):
            img_crop = imgs[:, i*size:(i+1)*size, j*size:(j+1)*size, :]  # imgs.shape = (B, H, W, C)
            liste.append(img_crop)
    print (len(liste))
    return  liste
    


if __name__ == "__main__" : 
    load_image("/home/jmerhrioui/soprasteria/semaine_2/data/dataset_y/images/train/train_image_6.png",  40)
