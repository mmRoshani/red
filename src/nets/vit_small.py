import timm
from torch import nn

class Vit_Small(nn.Module):
    def __init__(self, _number_of_classes: int, pretrained: bool = False):
        super(Vit_Small, self).__init__()

        self.vit = timm.create_model(
            'vit_small_patch16_224',
            pretrained=pretrained,
            img_size=32,
            patch_size=4,
            num_classes=_number_of_classes,
            in_chans=3
        )

    def forward(self, x):
        return self.vit(x)
