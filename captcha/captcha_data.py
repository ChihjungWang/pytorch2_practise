from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from captcha.image import ImageCaptcha
from random import randint, seed
import matplotlib.pyplot as plt
from tqdm import tqdm
import os.path as osp
from PIL import Image

char_list = [
    "0","1","2","3","4","5","6","7","8","9",
    "a","b","c","d","e","f","g","h","i","j","k","l",
    "m","n","o","p","q","r","s","t","u","v","w","x",
    "y","z"
]

class CaptchaData(Dataset):
    def __init__(self, path, char_list):
        super(CaptchaData, self).__init__()
        self.path = path
        self.data_list = []
        # 字元串列
        self.char_list = char_list
        # 字元轉 id
        self.char2index={
            self.char_list[i]: i for i in range(len(self.char_list))
        }
        with open(osp.join(path, 'label.txt'), 'r') as f:
            for line in f:
                line_data = line.strip('\n').split(',')
                self.data_list.append(line_data)
        
    def __getitem__(self, index):
        # 透過 index 去除驗證碼和對應的標籤
        chars = self.data_list[index][1]
        image = Image.open(osp.join(self.path, self.data_list[index][0]))

        # 將字元轉成 Tensor
        chars_tensor = self._numerical(chars)
        image_tensor = self._totensor(image)
        # 把標籤轉化為 onehot 編碼，以適應多標籤損失函數的輸入
        label = chars_tensor.long().unsqueeze(1)
        label_onehot = torch.zeros(4, 36)
        label_onehot.scatter_(1, label, 1)
        label = label_onehot.view(-1)
        return image_tensor, label
    
    def _numerical(self, chars):
        # 標籤字元轉 id
        chars_tensor = torch.zeros(4)
        for i in range(len(chars)):
            chars_tensor[i] = self.char2index[chars[i]]
        return chars_tensor
    
    def _totensor(self, image):
        # 圖片轉 Tensor
        return transforms.ToTensor()(image)
    
    def __len__(self):
        return len(self.data_list)
    

train_path = "F:/GitHub/pytorch2_practise/datasets/captcha/train"
data = CaptchaData(train_path, char_list)
dataloader = DataLoader(data, batch_size=128, shuffle=True)

val_path = "F:/GitHub/pytorch2_practise/datasets/captcha/val"
val_data = CaptchaData(val_path, char_list)
val_loader = DataLoader(val_data, batch_size=256, shuffle=True)


if __name__ == "__main__":
    img, label = data[10]
    predict = torch.argmax(label.view(-1, 36), dim=1)
    plt.title("-".join([char_list[lab.int()] for lab in predict]))
    plt.imshow(transforms.ToPILImage()(img))
    plt.show()


                       