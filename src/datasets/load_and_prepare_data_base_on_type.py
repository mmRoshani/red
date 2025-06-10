from src.constants.datasets_constants import DATA_SET_BBC, DATA_SET_SHAKESPEARE
from .image.load_and_prepare_data import load_and_prepare_data
from src.datasets.text.bbc.preprocess import load_and_tokenize
from src.constants import framework


def determine_dataset_type(config, log):
    train_loaders, test_loaders = None, None

    if config.DATASET_TYPE == DATA_SET_BBC:
        train_loaders, test_loaders = load_and_tokenize(framework.DATA_PATH, config)  # todo I dont know what is log for
    elif config.DATASET_TYPE == DATA_SET_SHAKESPEARE:
        raise ValueError("not implimented")
    else:
        train_loaders, test_loaders = load_and_prepare_data(config, log)

    return train_loaders, test_loaders