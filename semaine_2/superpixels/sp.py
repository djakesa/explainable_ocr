import numpy as np
from skimage.segmentation import slic, mark_boundaries
from skimage.util import img_as_float
from skimage.transform import resize
import torch
import os, sys
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm


class superpixels:
    def __init__(self, batch_size, n_segments, compactness):
        """
        batch_size: nombre d'images à récupérer
        n_segments: nombre de superpixels SLIC
        compactness: compacité SLIC
        """
        self.batch_size = batch_size
        self.n_segments = n_segments
        self.compactness = compactness
    
    #def load_image_single(self,path):
        #charger une seule image 
        #image_path = path
        #img = Image.open(image_path).convert("RGB")
        #img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        #imgs = img_tensor.numpy().transpose(0, 2, 3, 1)
        #return  imgs 
    
    def load_image(self,size):
        #charger une seule image 
        path = "/home/jmerhrioui/soprasteria/semaine_2/data/dataset_y/images/validation/validation_image_4.png"
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
        ############################################## On affiche image de base découpée en n*m ############################
        if imgs.ndim == 4 and imgs.shape[0] == 1:
            imgs = imgs.squeeze(0)

        H, W, _ = imgs.shape

        # carrés réguliers
        seg = np.zeros((H, W), dtype=int)
        sp_id = 0
        for i in range(0, H, size):
            for j in range(0, W, size):
                seg[i:i+size, j:j+size] = sp_id
                sp_id += 1

        # Ajoute les contours rouges comme avec mark_boundaries
        img_base_striee = mark_boundaries(imgs, seg, color=(1, 0, 0), mode='subpixel')

        return  liste, imgs, img_base_striee

    def superpixels(self):
        nbre_solo = 1 # pas de superpixels si patch unicolore
        liste, _ , _ = self.load_image(40)
        visus = []
        segs = []
        borders = []
        print("test_entrée")
        for e in tqdm(liste):
            x = e
            img = x
            img = img_as_float(img).squeeze()  # Supprimer dimensions inutiles
            if np.all(img == img[0, 0, :]): # Compare pixel un avec tout les autres
            #if np.std(img) < 1e-6:  # Vérifie si l'écart type est proche de zéro
                seg = slic(
                img,
                n_segments=nbre_solo,
                compactness=self.compactness,
                sigma=0,
                start_label=0
                )
            else  :
                seg = slic(
                    img,
                    n_segments=self.n_segments,
                    compactness=self.compactness,
                    sigma=0,
                    start_label=0
                )
            vis = mark_boundaries(img, seg, color=(1, 0, 0), mode='subpixel')
            visus.append(img)
            segs.append(seg)
            borders.append(vis)

        print("taillex", len(visus))
        return visus, segs, borders
    

    def save_superpixels(self, out_path="/home/jmerhrioui/soprasteria/semaine_2/test/superpixels_output.png"):
        visus, _, borders = self.superpixels()
        _, img , _= self.load_image(40)
        print("img shape_save", img.shape)  # (1, 1080, 1920, 3)
        
        # Get dimensions correctly from img shape
        h = img.shape[1]  # height = 1080
        w = img.shape[2]  # width = 1920
        print(f"Image size: {w}x{h}")
        
        # Create a new image with the correct dimensions
        output_img = Image.new('RGB', (w, h))
        
        # Calculate grid dimensions
        grid_size = 40
        rows = h // grid_size
        cols = w // grid_size
        
        # Paste each border image in the correct position
        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                if idx < len(borders):
                    arr = np.squeeze(borders[idx])  # retire les dims inutiles
                    arr = (arr * 255).astype(np.uint8)
                    if arr.shape[:2] != (grid_size, grid_size):
                        arr = np.array(Image.fromarray(arr).resize((grid_size, grid_size)))

                    border_img = Image.fromarray(arr)
                    output_img.paste(border_img, (j * grid_size, i * grid_size))
        
        # Save the final image
        output_img.save(out_path)
        print(f"Saved superpixels image to {out_path}")



