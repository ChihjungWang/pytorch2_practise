import torch
from torch import nn, optim
from captcha_model_resnet import resnet18_class
from captcha_data import dataloader, val_loader
from tqdm import tqdm
import os.path as osp

device = torch.device("cuda:0")
num_classes = 4*36
net = resnet18_class(num_classes=num_classes).to(device)

checkpoint_folder = 'F:/GitHub/pytorch2_practise/captcha/checkpoints'
model_name = 'captcha_resnet18_v01'
pth_file = osp.join(checkpoint_folder, model_name + ".pth")
if osp.exists(pth_file):
    net.load_state_dict(
        torch.load(pth_file)
        )
    print("模型已載入")

'''
epoch_lr = [
    (500, 0.01),
    (100, 0.001),
    (100, 0.001),
    (100, 0.0001)
]
'''

epoch_lr = [
    (100, 0.1),
    (50, 0.01),
    (50, 0.001)
]


criteron = nn.MultiLabelSoftMarginLoss()

def train():
    best_loss = 1e9
    net.to(device)
    accuracies = []
    losses = []
    val_accuracies =[]
    val_losses = []
    for n, (num_epoch, lr) in enumerate(epoch_lr):
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
        # optimizer = torch.optim.AdamW(net.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.1, amsgrad=False)
        # 優化器也可以多嘗試一下，一般來說使用 SGD 的對應學習率會比 adam 大一個數量級
        for epoch in range(num_epoch):
            # 每次驗證都會切換成 eval 模式，所以這裡要切換回來
            net.train()
            epoch_loss = 0.0
            epoch_acc = 0.0
            for i, (img, label) in tqdm(enumerate(dataloader)):
                out = net(img.to(device))
                label = label.to(device)
                # 清空 net 裡面所有參數的梯度
                optimizer.zero_grad()
                # 計算預測值與目標值之間的損失
                loss = criteron(out, label.to(device))
                # 計算梯度
                loss.backward()
                # 清空 net 裡面所有參數的梯度
                optimizer.step()
                # 整理輸出，方便與標籤進行比對
                predict = torch.argmax(out.view(-1, 36), dim=1)
                true_label = torch.argmax(label.view(-1, 36), dim=1)
                epoch_acc += torch.sum(predict == true_label).item()
                epoch_loss += loss.item()
            # 每訓練 3 次 驗證一次
            if epoch % 3 == 0:
                # no_grad 模式不計算梯度，可以執行得快一點
                with torch.no_grad():
                    net.eval()
                    val_loss = 0.0
                    val_acc = 0.0
                    for i, (img, label) in tqdm(enumerate(val_loader)):
                        out = net(img.to(device))
                        label = label.to(device)
                        loss = criteron(out, label.to(device))
                        predict = torch.argmax(out.view(-1,36), dim=1)
                        true_label = torch.argmax(label.view(-1, 36), dim=1)
                        val_acc += torch.sum(predict == true_label).item()
                        val_loss += loss.item()
                
                val_acc /= len(val_loader.dataset) *4
                val_loss /= len(val_loader)
                '''
                if val_loss < best_loss:
                    best_loss = val_loss
                    print('保存模型')
                    torch.save(
                        net.state_dict(),
                        osp.join(checkpoint_folder, model_name + ".pth")
                    )
                '''
            epoch_acc /= len(dataloader.dataset) *4
            epoch_loss /= len(dataloader)
            print(
                "epoch : {} , epoch loss : {} , epoch accuracy : {}".format(
                epoch + sum([e[0] for e in epoch_lr[:n]]), epoch_loss, epoch_acc 
                )
            )
            # 每遍歷 3 次資料集列印 1 次損失和準確率
            if epoch % 3 == 0:
                print(
                    "epoch : {} , val loss : {} , val accuracy : {}".format(
                    epoch + sum([e[0] for e in epoch_lr[:n]]), val_loss, val_acc 
                    )
                )
                # 紀錄損失和準確率
                for i in range(3):
                    val_accuracies.append(val_acc)
                    val_losses.append(val_loss)

            if epoch % 50 == 0:
                current_epoch_now = epoch + sum([e[0] for e in epoch_lr[:n]])
                model_name_save = model_name + '_' + str(current_epoch_now)
                pth_file = osp.join(checkpoint_folder, model_name_save + '.pth')
                torch.save(
                        net.state_dict(),
                        pth_file
                    )
                print('保存模型')

            accuracies.append(epoch_acc)
            losses.append(epoch_loss)


if __name__ == "__main__":
    train()










