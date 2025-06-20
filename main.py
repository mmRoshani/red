import os
from core.clustering.data_driven import calculate_label_distribution
from datasets.load_and_prepare_data_base_on_type import determine_dataset_type
from schemas import schema_factory
from utils.client_ids_list import client_ids_list_generator
from utils.framework_setup import FrameworkSetup
from utils.log import Log
from utils.log_path import log_path
from utils.variable_name import var_name
from utils.yaml_loader import load_objectified_yaml
from utils.checker import none_checker
import typer
from validators.config_validator import ConfigValidator
from validators.runtime_config import RuntimeConfig



def main(config_yaml_path: str = "./config.yaml"):
    config_yaml_path = none_checker(config_yaml_path, var_name(config_yaml_path))

    config_dict = load_objectified_yaml(config_yaml_path)

    config_dict = config_dict | {"desired_distribution": None}  # TODO: update

    config = ConfigValidator(**config_dict)

    log = Log(log_path(
        model_type=config.MODEL_TYPE,
        dataset_type=config.DATASET_TYPE,
        data_distribution=config.DATA_DISTRIBUTION,
        distance_metric=config.DISTANCE_METRIC,
        sensitivity_percentage=config.SENSITIVITY_PERCENTAGE,
        fed_avg=config.FED_AVG,
        dynamic_sensitivity_percentage=config.DYNAMIC_SENSITIVITY_PERCENTAGE,
        distance_metric_on_parameters=config.DISTANCE_METRIC_ON_PARAMETERS,
        pre_computed_data_driven_clustering=config.PRE_COMPUTED_DATA_DRIVEN_CLUSTERING,
        remove_common_ids=config.REMOVE_COMMON_IDS,
    ), config.MODEL_TYPE, config.DISTANCE_METRIC)

    # table_data = [[key, value] for key, value in config.items()] # TODO: fix items function in config validator
    # log.info(tabulate(table_data, headers=["Config Key", "Value"], tablefmt="grid"))


    log.info("----------    framework   setup   --------------------------------------------------")
    FrameworkSetup.path_setup(config)

    log.info("----------    datasets    distribution   --------------------------------------------------")
    train_loaders, test_loaders = determine_dataset_type(config, log)

    if config.PRE_COMPUTED_DATA_DRIVEN_CLUSTERING:
        log.info("clients train loader label distribution")
        # config = config | {"DATA_DRIVEN_CLUSTERING": compute_data_driven_clustering(train_loaders, config, log)}


    log.info("----------    runtime configurations  --------------------------------------------------")
    clients_id_list = client_ids_list_generator(config.NUMBER_OF_CLIENTS, log=log)

    config.RUNTIME_COMFIG = RuntimeConfig(
        clients_id_list=clients_id_list,
        train_loaders=train_loaders,
        test_loaders=test_loaders,
        log=log
    )

    log.info("----------    Schema  Factory --------------------------------------------------")
    schema_runner_fn = schema_factory(config.FEDERATED_LEARNING_SCHEMA, config.FEDERATED_LEARNING_TOPOLOGY ,log)
    schema_runner_fn(config, log)

if __name__ == "__main__":
    main()