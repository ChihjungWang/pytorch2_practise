from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from captcha.image import ImageCaptcha
from randon import randint, seed
import matplotlib.pyplot as plt
from tqdm import tqdm

char_list = [
    "0","1","2","3","4","5","6","7","8","9",
    "a","b","c","d","e","f","g","h","i","j","k","l",
    "m","n","o","p","q","r","s","t","u","v","w","x",
    ,"y","z"
]

class CaptchaData(Dataset):
    def __init__(self, char_list, num=10000):
        self.char_list = char_list
        self. char2index={
            self.char_list[i]: i for i in range(len(self.char_list))
        }