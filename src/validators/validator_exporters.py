from src.constants.models_constants import *
from src.constants.optimizer_constants import *
from typing import List
def model_constans_exporter() -> List[str]:
    return [
        MODEL_CNN,
        MODEL_LENET,
        MODEL_RESNET_18,
        MODEL_RESNET_50,
        MODEL_MOBILENET,
        MODEL_VGG,
        MODEL_VIT,
        MODEL_VIT_HYPER,
        MODEL_SWIN,
        MODEL_BERT,
        MODEL_SHAKESPEARE_HYPER,
        MODEL_ALBERT,
    ]

def optimizer_constans_exporter() -> List[str]:
    return [
        OPTIMIZER_SGD,
        OPTIMIZER_DSGD,
        OPTIMIZER_ADAM_W,
        OPTIMIZER_ADAM,
    ]

def model_constant_exporter_nlp()-> List[str]:
    return [MODEL_ALBERT, MODEL_SHAKESPEARE_HYPER, MODEL_BERT]

def model_constans_exporter_vision()-> List[str]:
    return [x for x in model_constans_exporter() if x not in [MODEL_ALBERT, MODEL_SHAKESPEARE_HYPER, MODEL_BERT]]



def transformer_model_size_exporter()->List[str]:
    return [TRANSFORMER_MODEL_SIZE_BASE,TRANSFORMER_MODEL_SIZE_LARGE]

