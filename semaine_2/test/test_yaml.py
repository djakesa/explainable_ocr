import sys
import os
import torch
#from torchsummary import summary
from torchinfo import summary
from ultralytics import YOLO
import cv2
from PIL import Image
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.import_data import ImportData


ImportData = ImportData()
#trainloader, test_loader = ImportData.load_data_mnist(batch_size=32)
#trainloader, test_loader = ImportData.load_data(batch_size=32)
#test_loader = ImportData.load_data_rica(batch_size=3)
path = ImportData.make_yaml(batch_size= 32)
print (path)
