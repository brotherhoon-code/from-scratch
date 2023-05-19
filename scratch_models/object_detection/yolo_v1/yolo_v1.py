import torch
import torch.nn as nn
import torch.nn.functional as F
from darknet import DarkNet
from torchsummary import summary


class YOLOv1(nn.Module):
    def __init__(self, features, num_bboxes=2, num_classes=20, bn=True):
        super(YOLOv1, self).__init__()

        self.feature_size = 7
        self.num_bboxes = num_bboxes
        self.num_classes = num_classes

        self.features = features
        self.conv_layers = self._make_conv_layer(bn)
        self.fc_layers = self._make_fc_layers()

    def forward(self, x):
        S, B, C = self.feature_size, self.num_bboxes, self.num_classes
        x = self.features(x)
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        x = x.view(-1, S, S, 5 * B + C)

        return x

    def _conv_bn_relu(self, bn=True, stride=1):
        layers = []
        layers.append(nn.Conv2d(1024, 1024, 3, stride=stride, padding=1))
        if bn:
            layers.append(nn.BatchNorm2d(1024))
        layers.append(nn.LeakyReLU(0.1))
        return nn.Sequential(*layers)

    def _make_conv_layer(self, bn):
        layers = []
        layers.append(self._conv_bn_relu(bn, stride=1))
        layers.append(self._conv_bn_relu(bn, stride=2))
        layers.append(self._conv_bn_relu(bn, stride=1))
        layers.append(self._conv_bn_relu(bn, stride=1))
        return nn.Sequential(*layers)

    def _make_fc_layers(self):
        S, B, C = self.feature_size, self.num_bboxes, self.num_classes
        layers = [
            nn.Flatten(),
            nn.Linear(7 * 7 * 1024, 4096),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(4096, S * S * (5 * B + C)),
            nn.Sigmoid(),
        ]
        return nn.Sequential(*layers)


if __name__ == "__main__":
    M = YOLOv1(features=DarkNet(is_cls_head=False, bn=True).bbone)
    summary(M, (3, 448, 448), device="cpu")
