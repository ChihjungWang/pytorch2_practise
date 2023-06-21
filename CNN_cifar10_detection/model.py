import torch
from torch import nn
import torch.nn.functional as F

# 本檔案中包含VGG-11和ResNet-18兩種模型結構，在學習的過程中可以任選一個進行練習

class VGG(nn.Module):
    def __init__(self, cfg, num_classes=10):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg)
        self.classifier = nn.Linear(512, num_classes)

    # 根據 cfg 設定參數逐步疊加網路層
    def _make_layers(self, cfg):
        layers = []
        # 輸入通道，彩色圖片的通道數量是3
        in_channels = 3
        for x in cfg:
            # 如果 X==M，那麼增加一個最大池化層
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x),
                    nn.ReLU(inplace=True)
                ]
                in_channels = x
        # 加入平均池化
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # 計算特徵網路
        out = self.features(x)
        out = out.view(out.size(0),-1)
        out = self.classifier(out)
        return out


class BasicBlock(nn.Module):
    def __init__(self, in_channels, mid_channels, stride=1):
        """
        in_channel:輸入通道數
        mid_channel: 中間的輸出通道數
        """
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels = in_channels,
            out_channels = mid_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(
            mid_channels,
            mid_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(mid_channels)
        # 定義短接網路，如果不需要調整維度，shortcut 是一個空的 nn.Sequential
        self.shortcut = nn.Sequential()
        # 因為 shortcut 後需要將兩個分支累加，所以要求兩個分支的維度匹配
        # 所以 input_channels 與最終的 channels 不匹配時，需要透過 1x1 的卷積進行

        if stride != 1 or in_channels != mid_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                in_channels,
                mid_channels,
                kernel_size=1,
                stride=stride,
                bias=False
                ),
                nn.BatchNorm2d(mid_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        # 架設 basicblock
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # 最後的線性層
        self.linear = nn.Linear(512, num_classes)
    
    def _make_layer(self, block, mid_channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        # stride 僅指定第一個block的stride，後面的 stride都是1
        for stride in strides:
            layers.append(block(self.in_channels, mid_channels, stride))
            self.in_channels = mid_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# 建構 ResNet-18模型
def resnet18():
    return ResNet(BasicBlock,[2,2,2,2])

# 建構vgg-11模型
def vgg11():
    cfg = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]
    return VGG(cfg)

if __name__ == "__main__":
    from torchsummary import summary
    vggnet = vgg11().cuda()
    resnet = resnet18().cuda()
    summary(vggnet,(3,32,32))
    summary(resnet, (3,32,32))


# python model.py