from captcha_model_resnet import resnet18_class
from captcha_data import char_list
from torchvision import transforms
import torch
import os.path as osp
import gradio as gr


# 增加必要的圖片轉 Tensor 的方法
transform = transforms.Compose([
    # transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

device = torch.device("cuda:0")
num_classes = 4*36
net = resnet18_class(num_classes=num_classes).to(device)

checkpoint_folder = 'F:/GitHub/pytorch2_practise/captcha/checkpoints'
model_name = 'captcha_resnet18_v01_100'
pth_file = osp.join(checkpoint_folder, model_name + ".pth")
if osp.exists(pth_file):
    net.load_state_dict(
        torch.load(pth_file)
        )
    print("模型已載入")
net.eval()

def predict(img):
    img = transform(img).unsqueeze(0)
    img_tensor = img.to(device)
    output= net(img_tensor).view(4,36)
    with torch.no_grad():
        output = torch.argmax(output, dim=1)

    label = "".join([char_list[lab.int()] for lab in output])
    return label

gr.Interface(fn=predict, 
             inputs=gr.Image(type="pil"),
             outputs="label").launch()
