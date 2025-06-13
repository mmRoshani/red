import copy
import yaml
from collections import namedtuple
from typing import Dict, Any
from pathlib import Path
from src.constants.models_constants import TRANSFORMER_MODEL_SIZE_BASE
from src.constants.loss_constants import LOSS_CROSS_ENTROPY, LOSS_MASKED_CROSS_ENTROPY, LOSS_SMOOTHED_CROSS_ENTROPY
from src.constants.optimizer_constants import OPTIMIZER_ADAM


def yaml_to_object(data):
    if isinstance(data, dict):
        return namedtuple('YAMLObject', data.keys())(**{k: yaml_to_object(v) for k, v in data.items()})
    elif isinstance(data, list):
        return [yaml_to_object(item) for item in data]
    else:
        return data


def load_objectified_yaml(yaml_path: str):
    if not yaml_path:
        raise ValueError('yaml_path cannot be empty')

    with open(yaml_path, 'r') as file:
        config = yaml.safe_load(file)

    yaml_obj = yaml_to_object(config)

    config_dict = {
        'federated_learning_schema': None,
        'federated_learning_topology': None,
        'draw_topology': False,
        'client_k_neighbors': None,
        'client_role': None,
        'federation_id': "",
        "device": "cuda",
        "gpu_index": 0,
        "random_seed": 42,
        "learning_rate": "0.001",
        "model_type": None,
        "transformer_model_size": TRANSFORMER_MODEL_SIZE_BASE,
        "pretrained_models": False,
        "dataset_type": None,
        "loss_function": LOSS_CROSS_ENTROPY,
        "optimizer": OPTIMIZER_ADAM,
        "data_distribution_kind": None,
        "desired_distribution": None,
        "dirichlet_beta": 0.1,
        "distance_metric": None,
        "dynamic_sensitivity_percentage": True,
        "sensitivity_percentage": None,
        "remove_common_ids": False,
        "fed_avg": False,
        "chunking": False,
        "chunking_with_gradients": True,
        "chunking_parts": 5.0,
        "chunking_random_section": False,
        "aggregation_strategy": None,
        "aggregation_sample_scaling": False,
        "distance_metric_on_parameters": False,
        "number_of_epochs": None,
        "train_batch_size": None,
        "test_batch_size": None,
        "transform_input_size": None,
        "weight_decay": 1e-4,
        "number_of_clients": 10,
        "client_sampling_rate": 1.0,
        "pre_computed_data_driven_clustering": False,
        "do_cluster": True,
        "clustering_period": 6,
        "federated_learning_rounds": 6,
        "stop_avg_accuracy": None,
        "save_before_aggregation_models": False,
        "save_global_models": False,
        "mean_accuracy_to_csv": True,
        "use_global_accuracy_for_noniid": True,

    }

    with open(yaml_path, "r") as f:
        yaml_config = yaml.safe_load(f)

    if yaml_config:
        config_dict.update(yaml_config)

    return config_dict
