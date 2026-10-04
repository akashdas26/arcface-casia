"""IResNet (ArcFace-style improved ResNet) backbone."""
import torch
import torch.nn as nn


class IBasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-5)
        self.conv1 = nn.Conv2d(inplanes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-5)
        self.prelu = nn.PReLU(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-5)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv2(self.prelu(self.bn2(self.conv1(self.bn1(x)))))
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        return out + identity


class IResNet(nn.Module):
    def __init__(self, layers, embedding_size=512, dropout=0.4):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-5)
        self.prelu = nn.PReLU(64)
        self.layer1 = self._make(64, layers[0], 2)
        self.layer2 = self._make(128, layers[1], 2)
        self.layer3 = self._make(256, layers[2], 2)
        self.layer4 = self._make(512, layers[3], 2)
        self.bn2 = nn.BatchNorm2d(512, eps=1e-5)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(512 * 7 * 7, embedding_size)          # 112 -> 7 after 4 stride-2 stages
        self.features = nn.BatchNorm1d(embedding_size, eps=1e-5)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def _make(self, planes, blocks, stride):
        ds = nn.Sequential(nn.Conv2d(self.inplanes, planes, 1, stride, bias=False),
                           nn.BatchNorm2d(planes, eps=1e-5))
        layers = [IBasicBlock(self.inplanes, planes, stride, ds)]
        self.inplanes = planes
        layers += [IBasicBlock(planes, planes) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.prelu(self.bn1(self.conv1(x)))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.dropout(self.bn2(x))
        return self.features(self.fc(torch.flatten(x, 1)))


def build_backbone(name, embedding_size=512, dropout=0.4):
    cfg = {"iresnet50": [3, 4, 14, 3], "iresnet100": [3, 13, 30, 3]}
    return IResNet(cfg[name], embedding_size, dropout)
