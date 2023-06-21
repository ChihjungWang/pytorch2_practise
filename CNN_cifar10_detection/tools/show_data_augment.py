import torchvision
from torchvision import transforms
import torch
from torchvision.utils import make_grid

import matplotlib.pyplot as plt
import sys

# 加入目錄
root = "F:/GitHub/pytorch2_practise/CNN_cifar10_detection"
sys.path.append(root)
from config import data_folder

def show_batch(display_transform=None):
    # 重新定義一個不帶 normalize 的 dataloader ,因為歸一化處理化的圖片很難辨認
    if display_transform is None:
        display_transform = transforms.ToTensor()
    display_set = torchvision.datasets.CIFAR10(
        root=data_folder, train=True, download=True, transform=display_transform 
    )
    display_loader = torch.utils.data.DataLoader(display_set, batch_size=32)
    topil = transforms.ToPILImage()
    # dataloader 物件無法直接取 index，可以透過這種方式取其中的元素
    for batch_img, batch_lable in display_loader:
        # 建立Tensor
        grid = make_grid(batch_img, nrow=8)
        # 轉成圖形
        grid_img = topil(grid)
        plt.figure(figsize=(15,15))
        plt.imshow(grid_img)
        grid_img.save(root + "/img/trans_cifar10.png")
        plt.show()
        break

if __name__ == "__main__":
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

show_batch(transform_train)
# cd F:\GitHub\pytorch2_practise\CNN_cifar10_detection\tools\
# python tools\show_data_augment.py