from torch import nn
import torchvision.models as models


class VGG_16(nn.Module):
    def __init__(self, _number_of_classes: int, pretrained: bool = False):
        super(VGG_16, self).__init__()

        self.vgg16 = models.vgg16(weights=models.VGG16_Weights if pretrained else None)
        self.vgg16.avgpool = nn.AdaptiveAvgPool2d(1)
        self.vgg16.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, _number_of_classes),
        )

    def forward(self, x):
        return self.vgg16(x)
