from torchvision import models
import torchvision
from torch import nn, optim
import torch
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # 使用 nn.Sequential 架設子模組
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Dropout(0.5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        # 全連接子模組
        self.fc = nn.Sequential(
            nn.Linear(20*7*64,1024),
            nn.Dropout(0.5),
            nn.ReLU()
        )
        # 輸出層
        self.rfc = nn.Sequential(nn.Linear(1024, 4*36))

    def forward(self, x):
        out = torchvision.transforms.Grayscale(x)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        out = self.rfc(out)
        return out

# net = CNN()


class resnet18_class(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(resnet18_class, self).__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.net = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.fcs = torch.nn.Sequential(
            torch.nn.Linear(512, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        x = self.net(x)
        x = x.view(x.shape[0], -1)
        return self.fcs(x)

