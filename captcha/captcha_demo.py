from captcha_model import net
from captcha_data import val_data, char_list
from captcha_train import device
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
import os.path as osp


net.to(device)
checkpoint_folder = 'F:/GitHub/pytorch2_practise/captcha/checkpoints'
model_name = 'captcha_v01'
pth_path = osp.join(checkpoint_folder, model_name + ".pth")
net.load_state_dict(torch.load(pth_path))
print('模型已載入')
net.eval()
img, label = val_data[12]
prediction = net(img.unsqueeze(0).to(device)).view(4,36)
predict = torch.argmax(prediction, dim=1)

print(
    "Predict Label: {}".format(
        "-".join([char_list[lab.int()] for lab in predict])
    )
)

plt.imshow(transforms.ToPILImage()(img))
plt.show()