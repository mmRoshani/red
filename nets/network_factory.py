import torchvision.models as torchModelType

from constants.models_constants import MODEL_CNN, MODEL_RESNET_18, MODEL_RESNET_50, MODEL_MOBILENET, MODEL_VGG, \
    MODEL_VIT, MODEL_SWIN, MODEL_LENET
from .cnn import CNN
from .lenet import LeNet
from .resnet_18 import ResNet_18
from .resnet_50 import ResNet_50
from .mobilenet_v2 import MobileNetV2
from .swin_base import Swin_base
from .vgg_16 import VGG_16
from .vit_small import Vit_Small


def network_factory(model_type: str, number_of_classes: int, pretrained: bool) -> torchModelType:
    if model_type == MODEL_CNN:
        return CNN(number_of_classes)
    elif model_type == MODEL_LENET:
        return LeNet(number_of_classes)
    elif model_type == MODEL_RESNET_18:
        return ResNet_18(number_of_classes, pretrained)
    elif model_type == MODEL_RESNET_50:
        return ResNet_50(number_of_classes, pretrained)
    elif model_type == MODEL_MOBILENET:
        return MobileNetV2(number_of_classes, pretrained)
    elif model_type == MODEL_VGG:
        return VGG_16(number_of_classes, pretrained)
    elif model_type == MODEL_VIT:
        return Vit_Small(number_of_classes, pretrained)
    elif model_type == MODEL_SWIN:
        return Swin_base(number_of_classes, pretrained)



