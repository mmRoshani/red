import numpy as np

from torch.utils.data import Subset


def get_labels(dataset):
    """
    Helper function to extract labels from a datasets or subset.
    Supports datasets with .targets or .target attributes.
    For Subset, extracts labels from underlying datasets using indices.
    """
    if isinstance(dataset, Subset):
        # Access underlying datasets and indices
        base_dataset = dataset.dataset
        indices = dataset.indices

        # Try to get labels from base_dataset
        if hasattr(base_dataset, 'targets'):
            labels = np.array(base_dataset.targets)[indices]
        elif hasattr(base_dataset, 'target'):
            labels = np.array(base_dataset.target)[indices]
        else:
            raise AttributeError("Underlying datasets has no attribute 'targets' or 'target'")
    else:
        # Dataset is not a Subset
        if hasattr(dataset, 'targets'):
            labels = np.array(dataset.targets)
        elif hasattr(dataset, 'target'):
            labels = np.array(dataset.target)
        else:
            raise AttributeError("Dataset has no attribute 'targets' or 'target'")
    return labels


def check_train_test_class_mismatch(train_ds, test_ds):
    train_labels = get_labels(train_ds)
    test_labels = get_labels(test_ds)

    train_classes = set(np.unique(train_labels))
    test_classes = set(np.unique(test_labels))

    mismatch_classes = test_classes - train_classes

    if mismatch_classes:
        print(f"Mismatch detected! Test classes not in train: {mismatch_classes}")
    else:
        print("No mismatch: test classes are all present in training data.")

    return mismatch_classes