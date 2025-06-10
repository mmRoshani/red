from torch import nn
import torchvision.models as models


class ResNet_18(nn.Module):
    def __init__(self, _number_of_classes: int, pretrained: bool = False):
        super(ResNet_18, self).__init__()

        assert _number_of_classes > 1

        self.resnet = models.resnet18(weights=models.ResNet18_Weights if pretrained else None)
        self.resnet.fc = nn.Sequential(nn.Linear(512, _number_of_classes))

    def forward(self, x):
        return self.resnet(x)
