from src.constants.datasets_constants import DATA_SET_BBC, DATA_SET_SHAKESPEARE , DATA_SET_YAHOO
from .image.load_and_prepare_data import load_and_prepare_data 
from .text.bbc.preprocess import load_and_tokenize
from src.constants import framework
from .text.shakespeare.shakespear_factory import dl_generator_shakespeare
from .text.yahoo.yahoo_factory import yahoo_factory


def determine_dataset_type(config, log):

    train_loaders, test_loaders = None, None

    if config.DATASET_TYPE == DATA_SET_BBC:
        train_loaders, test_loaders = load_and_tokenize(data_dir=framework.DATA_PATH ,config=config, log=log)

    elif config.DATASET_TYPE == DATA_SET_YAHOO:
        yahoo_factory(framework.DATA_PATH, config)

    elif config.DATASET_TYPE == DATA_SET_SHAKESPEARE:
        train_loaders, test_loaders = dl_generator_shakespeare(
            config,
            dataset="shakespeare",# dataset name
            data_dir=framework.DATA_PATH,  # path to data (e.g., all_data/train)
            batch_size=config.BATCH_SIZE,
            chunk_len=config.CHUNK_LEN,  # needs to be added to config
            is_validation=config.IS_VALIDATION  # optional, default False
        )
    else:  
        train_loaders, test_loaders = load_and_prepare_data(config, log)

    return train_loaders, test_loaders