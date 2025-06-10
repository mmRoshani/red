from torch import nn
import torchvision.models as models

class ResNet_18(nn.Module):
    def __init__(self, _number_of_classes: int, pretrained: bool = False):
        super(ResNet_18, self).__init__()

        if pretrained:
            self.resnet = models.resnet18(weights="IMAGENET1K_V1")
            in_features = self.resnet.fc.in_features
            self.resnet.fc = nn.Linear(in_features, _number_of_classes)
        else:
            self.resnet = models.resnet18(num_classes=_number_of_classes)

    def forward(self, x):
        return self.resnet(x)