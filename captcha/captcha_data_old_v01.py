from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from captcha.image import ImageCaptcha
from random import randint, seed
import matplotlib.pyplot as plt
from tqdm import tqdm
import os.path as osp

char_list = [
    "0","1","2","3","4","5","6","7","8","9",
    "a","b","c","d","e","f","g","h","i","j","k","l",
    "m","n","o","p","q","r","s","t","u","v","w","x",
    "y","z"
]

class CaptchaData(Dataset):
    def __init__(self, char_list, num=10000):
        super(CaptchaData, self).__init__()
        # 字元串列
        self.char_list = char_list
        # 字元轉 id
        self.char2index={
            self.char_list[i]: i for i in range(len(self.char_list))
        }
        # 標籤串列
        self.label_list = []
        # 圖片串列
        self.img_list = []
        self.num = num
        for i in tqdm(range(self.num)):
            chars = ""
            for i in range(4):
                chars += self.char_list[randint(0, 35)]
            image = ImageCaptcha().generate_image(chars)
            self.img_list.append(image)
            # 不區分大小寫
            self.label_list.append(chars) #.lower()


    
    def __getitem__(self, index):
        # 透過 index 去除驗證碼和對應的標籤
        chars = self.label_list[index]
        # image = self.img_list[index].convert("L")
        image = self.img_list[index]
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
        # 必須指定 Dataset 的長度
        return self.num

# 實例化一個 dataset，大概要 10000 個樣本才能訓練出比較好的結果
data = CaptchaData(char_list, num=10000)
# num_worders 多處理程序載入
# dataloader = DataLoader(data, batch_size=128, shuffle=True, num_workers=4)
dataloader = DataLoader(data, batch_size=128, shuffle=True)
val_data = CaptchaData(char_list, num=2000)
# val_loader = DataLoader(val_data, batch_size=256, shuffle=True, num_workers=4)
val_loader = DataLoader(val_data, batch_size=256, shuffle=True)


if __name__ == "__main__":
    # 可以透過以下方式從資料集中獲取圖片的對應標籤
    '''
    img, label = data[10]
    predict = torch.argmax(label.view(-1, 36), dim=1)
    plt.title("-".join([char_list[lab.int()] for lab in predict]))
    plt.imshow(transforms.ToPILImage()(img))
    plt.show()
    '''
    root  = "F:/GitHub/pytorch2_practise/datasets/captcha"
    label_list = []
    for i, (img, label) in enumerate(val_data):
        pil_img = transforms.ToPILImage()(img)
        predict = torch.argmax(label.view(-1, 36), dim=1)
        predict_str = "".join([char_list[lab.int()] for lab in predict])
        img_name = str(i+1).zfill(4)+".jpg"
        label_list.append(img_name + " " + predict_str)
        pil_img.save(osp.join(root, 'val', img_name))


    label_txt = osp.join(root, 'val', "label.txt")
    with open(label_txt, 'w') as fp:
        for item in label_list:
            fp.write("%s\n" % item)
        print('Done')


                       