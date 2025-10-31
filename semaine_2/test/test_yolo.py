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
trainloader, test_loader = ImportData.load_data_mnist_rota(batch_size=32)

print(len(test_loader.dataset))

#-----------------------On entraine le modèle -----------------------
ENTRAINEMENT = False
donnees = "/home/jmerhrioui/soprasteria/semaine_2/data/chiffres_RICA"
save_dir='/home/jmerhrioui/soprasteria/semaine_2/predictions'
model_maison = True
if model_maison == True :
    # PATH = "/home/jmerhrioui/soprasteria/semaine_2/model/jhmi_yolov5_OCR_model.pt"
    # PATH = "/home/jmerhrioui/soprasteria/semaine_2/model/ods_yolov5_OCR_model.pt"
    PATH = "/home/jmerhrioui/soprasteria/semaine_2/model/best.pt"
if ENTRAINEMENT == True: 
    model = YOLO("yolov8n-cls.pt")
    summary(model)
    results = model.train(data=donnees, epochs=3)
    #results = model.val()

else:
    # Load YOLOv5 model
    model = torch.hub.load('/home/jmerhrioui/soprasteria/yolov5', 'custom', path=PATH, source='local')
    model.conf = 0.25 # Confidence threshold
    model.iou = 0.4 # NMS IoU threshold
    
    # Run inference
    #img_path = "/home/jmerhrioui/soprasteria/semaine_2/data/dataset_y/images/validation/validation_image_4.png"
    img_path = "/home/jmerhrioui/soprasteria/semaine_2/test/image_bruitee_exemple.png"
    results = model(img_path)  # Pass only the image path
    
    # Show results
    results.show()
    results.save(save_dir=save_dir)
    
    # Print model summary
    print("\nModel Summary:")
    summary(model)
    
    # Get predictions as pandas DataFrame
    df = results.pandas().xyxy[0]
    print("\nDetection Results:") 
    print(df["name"].to_string())
    print ("\nAll Results DataFrame:")
    print(df)


        
