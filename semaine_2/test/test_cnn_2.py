import torch
import matplotlib.pyplot as plt
import numpy as np
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.simple_cnn import SimpleCNN
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.import_data import ImportData
from train.train_cnn import CNN  # classe qui contient train() et evaluate()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.import_data import ImportData


ImportData = ImportData()
#trainloader, test_loader = ImportData.load_data_mnist(batch_size=32)
#test_loader = ImportData.load_data_rica(batch_size=32)
trainloader, test_loader = ImportData.load_data_mnist_rota(batch_size=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Charger saved model 
model = SimpleCNN().to(device)
model.load_state_dict(torch.load("/home/jmerhrioui/soprasteria/poc/train/cnn_mnist__rota_ckpt.pth", map_location=device))
model.eval()

test_loss, test_acc = CNN.evaluate(model, test_loader, device)
print(f"Test Loss : {test_loss:.4f}")
print(f"Test Accuracy : {test_acc:.2f}")

# --------------------------Ici on affiche les preds --------------------------
classes = test_loader.dataset.classes
print(classes)

x, y = next(iter(test_loader))
x = x.to(device)
print (x.shape) 
with torch.no_grad():
    logits = model(x)
pred = logits.argmax(1).cpu()

n_show = 32
fig, axs = plt.subplots(1, n_show, figsize=(16, 2))
for i in range(n_show):
    img = x[i].cpu() * 0.5 + 0.5  # dénormaliser
    axs[i].imshow(np.transpose(img.numpy(), (1, 2, 0)))
    axs[i].set_title(f"GT: {classes[y[i]]}\nP: {classes[pred[i]]}")
    axs[i].axis('off')

plt.tight_layout()
plt.savefig("/home/jmerhrioui/soprasteria/poc/test/test_img/predictions.png", dpi=300)  #  Sauvegarde image
plt.close(fig)
