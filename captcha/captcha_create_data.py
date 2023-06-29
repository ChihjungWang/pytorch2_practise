from captcha.image import ImageCaptcha
from random import randint
from tqdm import tqdm
import os.path as osp
import os


char_list = [
    "0","1","2","3","4","5","6","7","8","9",
    "a","b","c","d","e","f","g","h","i","j","k","l",
    "m","n","o","p","q","r","s","t","u","v","w","x",
    "y","z"
]


def createCaptchaImage(path, char_list, num=2000):
    if not osp.isdir(path):
        os.mkdir(path)
    path
    img_list = []
    label_list = []
    for i in tqdm(range(num)):
        chars = ""
        for _ in range(4):
            chars += char_list[randint(0, len(char_list)-1)]
        image = ImageCaptcha().generate_image(chars)
        img_list.append(image)
        img_name = str(i+1).zfill(4)+".jpg"
        label_list.append(img_name + "," + chars)
        image.save(osp.join(path, img_name))

    label_txt = osp.join(path,"label.txt")
    with open(label_txt, 'w') as fp:
        for item in label_list:
            fp.write("%s\n" % item)
        print('Done')


if __name__ == "__main__":
    train_path = "F:/GitHub/pytorch2_practise/datasets/captcha/train"
    val_path = "F:/GitHub/pytorch2_practise/datasets/captcha/val"
    createCaptchaImage(train_path, char_list, 10000)
    createCaptchaImage(val_path, char_list, 2000)