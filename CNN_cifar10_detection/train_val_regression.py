import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from lr_find import lr_find
from model import resnet18
from data import create_datasets
from config import data_folder, batch_size, device, epochs
from generate_data import BoxData
from train_val import train_val

# 修改 ResNet 的最後一層的輸出
net = resnet18()
net.linear = nn.Linear(in_features=512, out_features=4, bias=True)
# 載入並封裝 CIFAR-10 資料
train_loader, val_loader = create_datasets(data_folder)
# 將模型遷移到 GPU
net.to(device)
# 將 CIFAR-10 資料轉換成邊框檢測資料
traindata = BoxData(train_loader.dataset)
# trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True, num_workers = 4)
trainloader = DataLoader(traindata, batch_size=batch_size, shuffle=True)
# 以 L1Loss作為損失函數
criteron = nn.L1Loss()
# 載入驗證資料
valdata = BoxData(val_loader.dataset)
# valloader = DataLoader(valdata, batch_size=batch_size, shuffle=True, num_workers= 4)
valloader = DataLoader(valdata, batch_size=batch_size, shuffle=True)
#  預先進行學習率搜尋，根據曲線確定初始學習率
# best_lr = lr_find(net, optim.SGD, train_loader, criteron)
# print("best_lr", best_lr)
# 訓練模型
train_val(
    net, trainloader, valloader, criteron, epochs, device, model_name="reg"
)
