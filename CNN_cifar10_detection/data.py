import torchvision
from torchvision import transforms
import torch

from config import data_folder, batch_size

def create_datasets(data_folder, transform_train=None, transform_test=None):
    if transform_train is None:
        transform_train = transforms.Compose([
            # 擴張之後再隨機裁剪
            transforms.RandomCrop(32, padding=4),
            # 隨機翻轉
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # 像素平均差與方差進行歸一化
            transforms.Normalize((0.4914,0.4822,0.446),(0.2223,0.1994,0.2010))
        ])
    if transform_test is None:
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914,0.4822,0.446),(0.2223,0.1994,0.2010))
        ])
    # 訓練集
    trainset = torchvision.datasets.CIFAR10(
        root = data_folder, train=True, download=True, transform=transform_train
    )
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    # 測試集
    testset = torchvision.datasets.CIFAR10(
        root = data_folder, train=False, download=True, transform=transform_test
    )
    # testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

    return trainloader, testloader