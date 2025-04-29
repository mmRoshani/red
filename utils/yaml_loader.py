import yaml
from collections import namedtuple


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
        'model_type': getattr(yaml_obj, 'model_type', None),
        'dataset_type': getattr(yaml_obj, 'dataset_type', None),
        'data_distribution_kind': getattr(yaml_obj, 'data_distribution_kind', None),
        'distance_metric': getattr(yaml_obj, 'distance_metric', None),
        'number_of_epochs': getattr(yaml_obj, 'number_of_epochs', None),
        'sensitivity_percentage': getattr(yaml_obj, 'sensitivity_percentage', None),
        'dynamic_sensitivity_percentage': getattr(yaml_obj, 'dynamic_sensitivity_percentage', True),
        'train_batch_size': getattr(yaml_obj, 'train_batch_size', None),
        'test_batch_size': getattr(yaml_obj, 'test_batch_size', None),
        'transform_input_size': getattr(yaml_obj, 'transform_input_size', None),
        'weight_decay': getattr(yaml_obj, 'weight_decay', 1e-4),
        'number_of_clients': getattr(yaml_obj, 'number_of_clients', 10),
        'dirichlet_beta': getattr(yaml_obj, 'dirichlet_beta', 0.1),
        'save_before_aggregation_models': getattr(yaml_obj, 'save_before_aggregation_models', True),
        'save_global_models': getattr(yaml_obj, 'save_global_models', True),
        'do_cluster': getattr(yaml_obj, 'do_cluster', True),
        'clustering_period': getattr(yaml_obj, 'clustering_period', 6),
        'federated_learning_rounds': getattr(yaml_obj, 'federated_learning_rounds', 6),
        'gpu_index': getattr(yaml_obj, 'gpu_index', None),
        'device': getattr(yaml_obj, 'device', None),
        'stop_avg_accuracy': getattr(yaml_obj, 'stop_avg_accuracy', None),
        'remove_common_ids': getattr(yaml_obj, 'remove_common_ids', False),
        'fed_avg': getattr(yaml_obj, 'fed_avg', False),
        'pre_computed_data_driven_clustering': getattr(yaml_obj, 'pre_computed_data_driven_clustering', False),
        'distance_metric_on_parameters': getattr(yaml_obj, 'distance_metric_on_parameters', False),
        'pretrained_models': getattr(yaml_obj, 'pretrained_models', False),
    }

    return config_dict
