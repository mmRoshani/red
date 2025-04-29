from datetime import datetime
from constants.framework import LOG_PATH
from constants.distances_constants import DISTANCE_COORDINATE


def log_path(model_type,
             dataset_type,
             data_distribution,
             distance_metric,
             sensitivity_percentage: int | float | None = None,
             dynamic_sensitivity_percentage: bool = True,
             fed_avg: bool = False,
             distance_metric_on_parameters: bool = True,
             pre_computed_data_driven_clustering: bool = False,
             remove_common_ids: bool = False,
             ) -> str:
    log_name = f"Model={model_type}"
    log_name += f"-Dataset={dataset_type}"
    log_name += f"-dd_N={data_distribution}"

    if fed_avg:
        log_name += f"-FED_AVG"
    elif pre_computed_data_driven_clustering:
        log_name += f"-PRE_COMPUTED_DATA_DRIVEN_CLUSTERING"
    else:
        log_name += f"-DISTANCE_METRIC={distance_metric}"

        if distance_metric == DISTANCE_COORDINATE:

            if not dynamic_sensitivity_percentage:
                log_name += f"-SENSITIVITY_PERCENTAGE={sensitivity_percentage}"
            else:
                log_name += f"-DYNAMIC_SENSITIVITY_PERCENTAGE"

            if remove_common_ids:
                log_name += f"-REMOVE_COMMON_IDS"

        if distance_metric_on_parameters:
            log_name += f"-DISTANCE_METRIC_ON_PARAMETERS"
        else:
            log_name += f"-DISTANCE_METRIC_ON_GRADIENT"

    log_name += "-date=%Y-%m-%d_%H"

    return f"{LOG_PATH}/{model_type}/{distance_metric}/{datetime.now().strftime(log_name)}.log"
