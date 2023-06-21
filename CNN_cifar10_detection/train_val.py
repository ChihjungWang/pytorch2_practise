from torch import optim, nn
import torch
import os.path as osp
from tqdm import tqdm
from torch.utils.tensorboard import summaryWriter

from config import epochs, device, data_folder, epochs, checkpoint_folder
from data import create_datasets
from model import vgg11

def train_val(net, trainloader, valloader, criteron, epochs, device, model_name="cls"):
    best_ass = 0.0
    best_loss = 1e9
    writer = summaryWriter("log")
    if osp.exists(osp.join(checkpoint_folder, model_name + ".pth")):
        net.load_state_dict(
            torch.load(osp.join(checkpoint_folder, model_name + ".pth"))
        )
        print("模型已在入")
    for n, (num_epochs, lr) in enumerate(epochs):
        optimizer = optim.SGD(net.parameters(),lr=lr, weight_decay=5e4, momentum=0.9)
        for epoch in range(num_epochs):
            net.train()
            epoch_loss = 0.0
            epoch_acc = 0.0
            for i, (img, lable) in tqdm(enumerate(trainloader), total=len(trainloader)):
                img, label = img.to(device), lable.to(device)
                output = net(img)
                optimizer.zero_grad()
                loss = criteron(output, label)
                loss.backward()
                optimizer.step()
                if model_name == "cls":
                    pred = torch.argmax(output, dim=1)
                    acc = torch.sum(pred == label)
                    epoch_acc += acc.item()
                epoch_loss += loss.item() * img.shape[0]
            # 計算這個epoch的平均損失
            epoch_loss /= len(trainloader.dataset)
            if model_name == "cls":
                epoch_acc /= len(trainloader.dataset)
                print(
                    "epoch_acc:{:.8f} epoch accuracy : {:.8f}".format(epoch_loss,epoch_acc)
                )
                