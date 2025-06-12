from src.constants.datasets_constants import DATA_SET_SHAKESPEARE, DATA_SET_BBC, DATA_SET_YAHOO
from src.validators.config_validator import ConfigValidator
from .shakespeare_downloader import download_and_preprocess_shakespeare
from .bbc_downloader import download_and_prepare_bbc_dataset
from .yahoo_downloader import download_and_prepare_yahoo_dataset

def dataset_downloader_factory(config: 'ConfigValidator'):
    
    if config.DATASET_TYPE == DATA_SET_SHAKESPEARE:
        return download_and_preprocess_shakespeare
    elif config.DATASET_TYPE == DATA_SET_BBC:
        return download_and_prepare_bbc_dataset
    elif config.DATASET_TYPE == DATA_SET_YAHOO:
        return download_and_prepare_yahoo_dataset
    else:
        raise ValueError(f'Unexpected value for {config.DATASET_TYPE}')