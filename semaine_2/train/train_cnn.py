# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T
import torchvision
from torch.utils.data import DataLoader
import os
import sys
from collections import defaultdict
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.simple_cnn import SimpleCNN

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.import_data import ImportData  

class CNN():
    def __init__(self,model,loader,device,epochs,lr,batch_size,ckpt_path):
        self.model=model
        self.loader=loader
        self.device=device
        self.epochs=epochs
        self.lr=lr
        self.batch_size=batch_size
        self.ckpt_path=ckpt_path

    @torch.no_grad()
    def evaluate(model, loader, device):
        model.eval()
        criterion = nn.CrossEntropyLoss() # fct de perte = cross entropy

        total, correct, loss_sum = 0, 0, 0.0
        per_class_total = np.zeros(10, dtype=int)  # ICI 10 correspond au nombre de classes (0-9)
        per_class_correct = np.zeros(10, dtype=int)
        confusion = np.zeros((10, 10), dtype=int)  # [true, pred]

        for x, y in loader:
            x, y = x.to(device), y.to(device) # x l'image et y le label
            logits = model(x)
            loss = criterion(logits, y)
            loss_sum += loss.item() * y.size(0)

            pred = logits.argmax(1)

            # Global stats
            correct += (pred == y).sum().item()
            total += y.size(0)

            # Per-class stats
            for t, p in zip(y.cpu().numpy(), pred.cpu().numpy()):
                per_class_total[t] += 1
                if t == p:
                    per_class_correct[t] += 1
                confusion[t, p] += 1

        # ---- Affichage ----
        print("\n=== Résultats par classe ===")
        for cls in range(10):
            acc = 100 * per_class_correct[cls] / max(1, per_class_total[cls])
            print(f"Chiffre {cls} : {per_class_correct[cls]}/{per_class_total[cls]} corrects ({acc:.1f}%)")

        print("\n=== Confusions les plus fréquentes ===")
        for true in range(10):
            for pred in range(10):
                if true != pred and confusion[true, pred] > 0:
                    print(f"{true} → {pred} : {confusion[true, pred]} fois")

        return loss_sum / total, correct / total 
    
    def train(self):
        epochs=self.epochs
        lr=self.lr
        batch_size=self.batch_size
        ckpt_path=self.ckpt_path
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
        model = SimpleCNN().to(device)
        data = ImportData()
        trainloader, testloader = data.load_data_mnist_rota(batch_size)  # attention ici charge data
        #trainloader, testloader = data.load_data_mnist(batch_size)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        for epoch in range(1, epochs + 1):
            model.train()
            running_loss = 0.0
            for x, y in trainloader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            train_loss = running_loss / len(trainloader)
            val_loss, val_acc = CNN.evaluate(model, testloader, device)
            scheduler.step()

            print(f"[{epoch:02d}/{epochs}] train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  val_acc={val_acc*100:.2f}%")

        #torch.save(model.state_dict(), ckpt_path)
        #print(f" Modèle sauvegardé : {ckpt_path}")
        print ("model non sauvegardé")


