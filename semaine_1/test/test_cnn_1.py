import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.simple_cnn import SimpleCNN
import torch
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from train.train_cnn import CNN
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.import_data import ImportData


ImportData = ImportData()
#trainloader, test_loader = ImportData.load_data_mnist(batch_size=32)
#trainloader, test_loader = ImportData.load_data(batch_size=32)
#test_loader = ImportData.load_data_rica(batch_size=3)
trainloader, test_loader = ImportData.load_data_mnist_rota(batch_size=32)

print(len(test_loader.dataset))

#-----------------------On entraine le modèle -----------------------


model = SimpleCNN()
x = torch.randn(4, 3, 32, 32)
y = model(x)
print(y.shape)  # torch.Size([4, 10])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CNN= CNN(model=model,
          loader=trainloader,
          device=device,
          epochs=5,
          lr=1e-3,
          batch_size=32,
          ckpt_path="/home/jmerhrioui/soprasteria/poc/train/cnn_mnist__rota_ckpt.pth") # allez dans train changer load data
CNN.train()