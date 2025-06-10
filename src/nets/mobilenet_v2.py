from torch import nn
import torchvision.models as models

class MobileNetV2(nn.Module):
    def __init__(self, _number_of_classes: int, pretrained: bool = False):
        super(MobileNetV2, self).__init__()

        self.mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights if pretrained else None)
        num_features = self.mobilenet.classifier[-1].in_features
        self.mobilenet.classifier[-1] = nn.Linear(num_features, _number_of_classes)

    def forward(self, x):
        return self.mobilenet(x)
