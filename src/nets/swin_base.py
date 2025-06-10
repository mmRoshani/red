import timm
from torch import nn

class Swin_base(nn.Module):
    def __init__(self, _number_of_classes: int, pretrained: bool = False):
        super(Swin_base, self).__init__()

        self.swin = timm.create_model(
            'swin_base_patch4_window7_224',
            pretrained=pretrained,
            num_classes=_number_of_classes,
            img_size=64,
        )

    def forward(self, x):
        return self.swin(x)

