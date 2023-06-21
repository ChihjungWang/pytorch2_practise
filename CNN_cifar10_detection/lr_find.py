import torch
from torch.optim import SGD
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from config import device, data_folder
from model import vgg11
from data import create_datasets

def lr_find(net, optimizer_class, dataloader, criteron, lr_list=[1*10**(i/2) for i in range(-20,0)], show=False, test_time=10):
    """
    net: 模型
    optimizer_class: 優化器類別
    dataloader: 資料
    criteron: 損失函數
    lr_list: 學習率列表
    show: 是否顯示結果
    test_time: 實驗次數
    """
    params = net.state_dict().copy()
    loss_matrix =[]
    for i, (img, label) in enumerate(dataloader):
        img, label = img.to(device), label.to(device)
        loss_list = []
        for lr in tqdm(lr_list):
            # 重新載入原始參數
            net.load_state_dict(params)
            # 訓練模型
            out = net(img)
            optimizer = optimizer_class(
                net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4
            )
            loss = criteron(out, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 計算更新模型之後的損失
            new_out = net(img)
            new_loss = criteron(new_out, label)
            loss_list.append(new_loss.item())
        
        loss_matrix.append(loss_list)
        if i + 1 == test_time:
            break
    loss_matrix = np.array(loss_matrix)
    loss_matrix = np.mean(loss_matrix, axis=0)
    if show:
        plt.plot([np.log10(lr) for lr in lr_list], loss_matrix)
        plt.savefig("img/lr_find.jpg")
        plt.show()
    # 計算損失下降幅度，尋找最佳學習率
    decrease = [
        loss_matrix[i+1]-loss_matrix[i] for i in range(len(lr_list)-1)
    ]

if __name__ == "__main__":
    net = vgg11().to(device)
    trainloader, _ = create_datasets(data_folder)
    criteron = CrossEntropyLoss()
    lr_list = [1*10**(i/3) for i in range(-30,0)]
    lr_find(net, SGD, trainloader, criteron, show=True)