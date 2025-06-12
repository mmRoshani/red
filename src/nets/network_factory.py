import torchvision.models as torchModelType

from src.constants.models_constants import (
    MODEL_CNN,
    MODEL_RESNET_18,
    MODEL_RESNET_50,
    MODEL_MOBILENET,
    MODEL_VGG,
    MODEL_VIT,
    MODEL_SWIN,
    MODEL_LENET,
    MODEL_VIT_HYPER,
    MODEL_SHAKESPEARE_HYPER,
    MODEL_LENET,
    MODEL_BERT,
    MODEL_ALBERT,
)
from .cnn import CNN
from .lenet import LeNet
from .resnet_18 import ResNet_18
from .resnet_50 import ResNet_50
from .mobilenet_v2 import MobileNetV2
from .shakes_hyper import ShakesHyper
from .swin_base import Swin_base
from .vgg_16 import VGG_16
from .vit_hyper import ViTHyper
from .vit_small import Vit_Small
from .bert import Bert

def network_factory(
    model_type: str, number_of_classes: int, pretrained: bool, random_seed=42
) -> torchModelType:
    model_map = {
        MODEL_CNN: lambda: CNN(number_of_classes),
        MODEL_LENET: lambda: LeNet(number_of_classes),
        MODEL_RESNET_18: lambda: ResNet_18(number_of_classes, pretrained),
        MODEL_RESNET_50: lambda: ResNet_50(number_of_classes, pretrained),
        MODEL_MOBILENET: lambda: MobileNetV2(number_of_classes, pretrained),
        MODEL_VGG: lambda: VGG_16(number_of_classes, pretrained),
        MODEL_VIT: lambda: Vit_Small(number_of_classes, pretrained),
        MODEL_VIT_HYPER: lambda: ViTHyper(number_of_classes, pretrained), # TODO: fix the input parameters
        MODEL_SWIN: lambda: Swin_base(number_of_classes, pretrained),
        MODEL_SHAKESPEARE_HYPER: lambda: ShakesHyper(number_of_classes, pretrained), # TODO: fix the input parameters
        MODEL_BERT: lambda: Bert(number_of_classes, pretrained),
        MODEL_ALBERT: lambda: Bert(number_of_classes, pretrained)
    }

    try:
        return model_map[model_type]()
    except KeyError:
        raise ValueError(f"Unknown model type: {model_type}")


