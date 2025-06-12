import numpy as np
from src.utils.log import Log
from src.utils.transform_array_to_binary import transform_array
from src.validators.config_validator import ConfigValidator
from src.constants.datasets_constants import (
    DATA_SET_BBC,
    DATA_SET_SHAKESPEARE,
)


def calculate_label_distribution(
    dataloader, loader_name: str, config: "ConfigValidator", log: "Log"
):
    number_of_classes = config.NUMBER_OF_CLASSES
    dataset_type = config.DATASET_TYPE

    label_counts = np.zeros(number_of_classes)

    if dataset_type in [
        DATA_SET_BBC,
        DATA_SET_SHAKESPEARE,
    ]:

        for batch in dataloader:
            labels = batch["labels"].numpy()
            for label in labels:
                label_counts[label] += 1

    else:
        label_counts = np.zeros(number_of_classes)
        for _, labels in dataloader:
            for label in labels.numpy():
                label_counts[label] += 1

    log.info(f"client {loader_name} label distribution is: {label_counts}")
    return label_counts, transform_array(label_counts)
