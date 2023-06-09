import torch
from torch.utils.data import Dataset
from torchvision import transforms
from numpy import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from data import create_datasets
from config import data_folder

def expand(img, background=(128,128,128), show=False):
    # 隨機將cifar10中的圖片貼到灰色背景中
    topil = transforms.ToPILImage()
    totensor = transforms.ToTensor()
    # 輸入的img是按 NCHW 形式排列的Tensor類型，需要先進行轉化
    img = np.array(topil(img)).astype(np.uint8)
    # 隨機生成貼上位置
    height, width, depth = img.shape
    ratio = random.uniform(1,2)
    # 左邊界位置
    left = random.uniform(0.3 * width, width * ratio - width)
    # 上邊界位置
    top = random.uniform(0.3 * height, height * ratio - height)
    while int(left + width) > int(width*ratio) or int(top + height) > int(height*ratio):
        ratio = random.uniform(1, 2)
        left = random.uniform(0.3* width, width * ratio - width)
        top = random.uniform(0.3* height, height * ratio - height)
    # 建立白色背景
    expand_img = np.zeros((int(height * ratio), int(width * ratio), depth), dtype = img.dtype)
    # 背景填充成灰色
    expand_img[:,:,:] = background
    # print(f'expand_img_size:{expand_img.shape}')
    # 將圖片按先前生成的隨機位置貼上到背景中
    expand_img[int(top):int(top+height), int(left):int(left+width)] = img

    # 展示圖片
    if show:
        expand_img_ = Image.fromarray(expand_img)
        draw = ImageDraw.ImageDraw(expand_img_)
        # 使用xmin, ymin, xmax, ymax座標繪製邊框
        draw.rectangle(
            [(int(left), int(top)), (int(left+width), int(top+height))],
            outline=(0,255,0),
            width = 2
        )
        # 保存圖片
        expand_img_.save("img/plane_bound_true.jpg")
        plt.subplot(121)
        plt.imshow(img)
        plt.subplot(122)
        plt.imshow(expand_img_)
        plt.savefig("img/expand_img.jpg")
        plt.show()
    
    # 紀錄圖片位置 (相對位置)
    xmin = left / (width * ratio)
    ymin = top / (height * ratio)
    xmax = (left + width) / (width * ratio)
    ymax = (top + height) / (height * ratio)

    #處理完之後還需要進行尺寸變化
    expand_img = totensor(
        Image.fromarray(expand_img).resize((32,32),Image.BILINEAR)
    )
    return expand_img, torch.Tensor([xmin, ymin, xmax, ymax])

#將生成方法些入Dataset中
class BoxData(Dataset):
    def __init__(self, dataset, show=False):
        super(BoxData, self).__init__()
        self.dataset = dataset
        # 用於展示
        self.show = show

    def __getitem__(self, index):
        img, lable = self.dataset[index]
        # print(f'img: {img}')
        # 使用線上生成的方式，將轉換函數加入Dataset中
        img, box = expand(img, show=self.show)
        return img, box
    
    def __len__(self):
        return len(self.dataset)
    
if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor()])
    train_loader, _ = create_datasets(data_folder, transform_train=transform)
    data = BoxData(train_loader.dataset, show=True)
    print(data[0][0].shape, data[0][1].shape)

# python generate_data.py