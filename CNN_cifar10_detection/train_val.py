from torch import optim, nn
import torch
import os.path as osp
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from config import epochs, device, data_folder, epochs, checkpoint_folder
from data import create_datasets
from model import vgg11

# 這裡為後續的回歸問題預留了一些程式
def train_val(net, trainloader, valloader, criteron, epochs, device, model_name="cls"):
    best_acc = 0.0
    best_loss = 1e9
    writer = SummaryWriter("log")
    # 如果模型檔案已經存在，先載入模型檔案再在此基礎上訓練
    if osp.exists(osp.join(checkpoint_folder, model_name + ".pth")):
        net.load_state_dict(
            torch.load(osp.join(checkpoint_folder, model_name + ".pth"))
        )
        print("模型已載入")
    for n, (num_epochs, lr) in enumerate(epochs):

        optimizer = optim.SGD(net.parameters(),lr=lr, weight_decay=5e-4, momentum=0.9)
        # 迴圈多次
        for epoch in range(num_epochs):
            print(f'第 {epoch} 個 epoch 開始訓練')
            net.train()
            epoch_loss = 0.0
            epoch_acc = 0.0
            # print(f'trainloader_size: {len(trainloader)}')
            # print(f'trainloader_enumerate: {trainloader_enumerate}')
            for i, (img, lable) in tqdm(enumerate(trainloader), total=len(trainloader)):
                # 將圖片與標籤都移動到GPU中
                img, label = img.to(device), lable.to(device)
                output = net(img)
                # 清空梯度
                optimizer.zero_grad()
                # 計算損失
                loss = criteron(output, label)
                # 反向傳播
                loss.backward()
                # 更新參數
                optimizer.step()
                # 分類問題容易使用準確率來衡量模型效果
                # 但是回歸模型無法按照分類問題的方法來計算準確率
                if model_name == "cls":
                    pred = torch.argmax(output, dim=1)
                    acc = torch.sum(pred == label)
                    # 累計準確率
                    epoch_acc += acc.item()
                # print(f'batch_size >> img.shape[0] : {img.shape[0]}')
                # print(f'loss.item(): {loss.item()}')
                # print(f'epoch_loss: {epoch_loss}')
                epoch_loss += loss.item() * img.shape[0]

            # 計算這個epoch的平均損失
            
            # print(f'trainloader_dataset: {len(trainloader.dataset)}')
            epoch_loss /= len(trainloader.dataset)
            if model_name == "cls":
                # 計算這個 epoch_acc 的平均準確率 
                epoch_acc /= len(trainloader.dataset)
                print(
                    "epoch_acc: {:.8f} epoch accuracy : {:.8f}".format(epoch_loss, epoch_acc)
                )
                # 將損失增加到 TensorBoard 中
                writer.add_scalar(
                    "epoch_loss_{}".format(model_name),epoch_loss,
                    sum([e[0] for e in epochs[:n]]) + epoch
                )
                # 將準確率增加到 TensorBoard 中
                writer.add_scalar(
                    "epoch_acc_{}".format(model_name), epoch_acc,
                    sum([e[0] for e in epochs[:n]]) + epoch
                )
            else:
                print("epoch loss: {:.8f}".format(epoch_loss))
                writer.add_scalar(
                    "epoch_loss_{}".format(model_name), epoch_loss,
                    sum([e[0] for e in epochs[:n]]) + epoch
                )
            # 無梯度模式下快速驗證
            with torch.no_grad():
                # 將 net 設定為驗證模式
                net.eval()
                val_loss = 0.0
                val_acc = 0.0
                for i, (img, lable) in tqdm(enumerate(valloader), total=len(valloader)):
                    img, label = img.to(device), lable.to(device)
                    output = net(img)
                    loss = criteron(output, label)
                    if model_name == "cls":
                        pred = torch.argmax(output, dim=1)
                        acc = torch.sum(pred == label)
                        val_acc += acc.item()
                    val_loss += loss.item() * img.shape[0]
                # 計算這個epoch的平均損失
                val_loss /= len(valloader.dataset)
                val_acc /= len(valloader.dataset)
                if model_name == "cls":
                    # 如果驗證之後的模型超過了目前最好的模型
                    if val_acc > best_acc:
                        # 更新 best_acc
                        best_acc = val_acc
                        # 保存模型
                        torch.save(
                            net.state_dict(),
                            osp.join(checkpoint_folder, model_name + ".pth")
                        )
                    print(
                        "validation loss: {:.8f} validation accuracy:{:.8f}".format(val_loss, val_acc)
                    )
                    writer.add_scalar(
                        "validation_loss_{}".format(model_name), val_loss, sum([e[0] for e in epoch[:n]]) + epoch
                    )
                else:
                    # 如果得到的損失比當前最好的損失還好
                    if val_loss < best_loss:
                        best_loss = val_loss
                        # 保存模型
                        torch.save(
                            net.state_dict(),
                            osp.join(checkpoint_folder, model_name + ".pth")
                        )
                    print("validation loss:{:8f}".format(val_loss))
                    writer.add_scalar(
                        "epoch_loss_{}".format(model_name), val_loss,sum([e[0] for e in epochs[:n]]) + epoch
                    )
    writer.close()


if __name__ == "__main__":
    trainloader, valloader = create_datasets(data_folder)