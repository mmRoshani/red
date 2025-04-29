import os
from constants.framework import MODELS_SAVING_PATH
from validators.config_validator import ConfigValidator


class FrameworkSetup:
    def __init__(self):
        pass

    @staticmethod
    def path_setup(config: 'ConfigValidator'):
        MODEL_SAVING_PATH = os.path.join(MODELS_SAVING_PATH, config.MODEL_TYPE, config.DATASET_TYPE) + "/"
        if not os.path.exists(MODEL_SAVING_PATH):
            os.makedirs(MODEL_SAVING_PATH)