from torchvision.models import resnet18
from torchvision import transforms, models
from PIL import Image
import torch
import os
from time import ctime
import gradio as gr

# 增加必要的圖片轉 Tensor 的方法
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class Model(torch.nn.Module):
    def __init__(self, num_classes=10):
        super(Model, self).__init__()
        resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        self.net = torch.nn.Sequential(*list(resnet.children())[:-1])
        self.fcs = torch.nn.Sequential(
            torch.nn.Linear(2048, 2048),
            torch.nn.BatchNorm1d(2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, num_classes)
        )
        
    def forward(self, x):
        x = self.net(x)
        x = x.view(x.shape[0], -1)
        return self.fcs(x)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 10
model = Model(num_classes).to(device)
model.load_state_dict(torch.load('checkpoints/10BigCats_sd_4.pth'))
model.eval()

class_names = ['AFRICAN LEOPARD','CARACAL','CHEETAH','CLOUDED LEOPARD','JAGUAR','LIONS', 'OCELOT', 'PUMA', 'SNOW LEOPARD', 'TIGER']

def predict(inp):
    inp = transform(inp).unsqueeze(0)
    img_tensor = inp.to(device)
    output= model(img_tensor)
    # print(f'output.shape : {output.shape}')
    # output.shape : torch.Size([1, 10]) 1行10列
    # 如果 torch.Size([10])是10行, 意義不一樣
    with torch.no_grad():
        prediction = torch.nn.functional.softmax(output[0], dim=0)
    confidences = {class_names[i]: float(prediction[i]) for i in range(len(class_names))} 
    return confidences

gr.Interface(fn=predict, 
             inputs=gr.Image(type="pil"),
             outputs=gr.Label(num_top_classes=5)).launch()