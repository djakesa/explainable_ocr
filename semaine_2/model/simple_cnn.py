import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__() # hérite de nn.Module
        
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(128 * 4 * 4, 128) 
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 32, 16x16
        x = self.pool(F.relu(self.conv2(x)))  # 64, 8x8
        x = self.pool(F.relu(self.conv3(x)))  # 128, 4x4
        x = x.view(x.size(0), -1)              # 128*4*4 = 2048
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x



