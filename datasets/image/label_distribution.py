import numpy as np
from utils.log import Log
from utils.transform_array_to_binary import transform_array


def calculate_label_distribution(
    dataloader, loader_name: str, number_of_classes, log: "Log"
):
    label_counts = np.zeros(number_of_classes)
    for _, labels in dataloader:
        for label in labels.numpy():
            label_counts[label] += 1

    log.info(f"client {loader_name} label distribution is: {label_counts}")

    return label_counts, transform_array(label_counts)
