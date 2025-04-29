import torch
from typing import Tuple

from constants.distances_constants import (
    DISTANCE_COORDINATE,
    DISTANCE_COSINE,
    DISTANCE_EUCLIDEAN,
)

from constants.models_constants import (
    MODEL_CNN,
    MODEL_RESNET_18,
    MODEL_RESNET_50,
    MODEL_MOBILENET,
    MODEL_VGG,
    MODEL_VIT,
    MODEL_SWIN
)

from constants.datasets_constants import (
    DATA_SET_STL_10,
    DATA_SET_CIFAR_10,
    DATA_SET_CIFAR_100,
    DATA_SET_FMNIST,
    DATA_SET_SVHN,
    DATA_SET_TINY_IMAGE_NET
)

from constants.data_distribution_constants import (
    DATA_DISTRIBUTION_N_20,
    DATA_DISTRIBUTION_N_30,
    DATA_DISTRIBUTION_DIR,
    DATA_DISTRIBUTION_FIX,
)
from utils.gpu_index_list import list_available_gpus


class ConfigValidator:
    def __init__(
            self,
            model_type: str,
            dataset_type: str,
            data_distribution_kind: str,
            distance_metric: str,
            number_of_epochs=None,
            sensitivity_percentage=None,
            dynamic_sensitivity_percentage: bool = True,
            train_batch_size=None,
            test_batch_size=None,
            transform_input_size=None,
            weight_decay=1e-4,
            number_of_clients=10,
            dirichlet_beta=0.1,
            save_before_aggregation_models: bool = False,
            save_global_models: bool = False,
            do_cluster: bool = True,
            clustering_period=6,
            federated_learning_rounds=6,
            desired_distribution=None,
            remove_common_ids: bool = False,
            gpu_index: int | None = None,
            device: str = None,
            fed_avg: bool = False,
            stop_avg_accuracy=None,
            pre_computed_data_driven_clustering: bool = False,
            distance_metric_on_parameters: bool = True,
            pretrained_models: bool = False,

    ):

        self.MODEL_TYPE = self._validate_model_type(model_type)
        self.DATASET_TYPE = self._validate_dataset_type(dataset_type)
        self.DATA_DISTRIBUTION = self._validate_data_distribution(data_distribution_kind, desired_distribution)
        self.DISTANCE_METRIC = self._set_distance_metric(distance_metric)
        self.NUMBER_OF_EPOCHS = self._set_number_of_epochs(number_of_epochs)
        self.DYNAMIC_SENSITIVITY_PERCENTAGE = dynamic_sensitivity_percentage
        self.SENSITIVITY_PERCENTAGE = self._set_sensitivity_percentage(sensitivity_percentage,
                                                                       dynamic_sensitivity_percentage)

        self.TRAIN_BATCH_SIZE, self.TEST_BATCH_SIZE, self.TRANSFORM_INPUT_SIZE = (
            self._set_transformer(
                train_batch_size,
                test_batch_size,
                transform_input_size,
            )
        )


        self.WEIGHT_DECAY = weight_decay
        self.NUMBER_OF_CLIENTS = number_of_clients
        self.NUMBER_OF_CLASSES = self._dataset_number_of_classes
        self.DIRICHLET_BETA = dirichlet_beta
        self.DESIRED_DISTRIBUTION = desired_distribution
        self.SAVE_BEFORE_AGGREGATION_MODELS = save_before_aggregation_models
        self.SAVE_GLOBAL_MODELS = save_global_models
        self.DO_CLUSTER = do_cluster
        self.CLUSTERING_PERIOD = clustering_period
        self.FEDERATED_LEARNING_ROUNDS = federated_learning_rounds
        self.REMOVE_COMMON_IDS = remove_common_ids
        self.GPU_INDEX = gpu_index
        self.DEVICE = self._device(device, gpu_index)
        self.FED_AVG = fed_avg
        self.STOP_AVG_ACCURACY = self._stop_avg_accuracy(stop_avg_accuracy)
        self.PRE_COMPUTED_DATA_DRIVEN_CLUSTERING = pre_computed_data_driven_clustering
        self.DISTANCE_METRIC_ON_PARAMETERS = distance_metric_on_parameters
        self.PRETRAINED_MODELS = pretrained_models

    # def get_config(self):
    #
    #     config_dic = {
    #         "MODEL_TYPE": self.MODEL_TYPE,
    #         "DATASET_TYPE": self.DATASET_TYPE,
    #         "NUMBER_OF_CLASSES": self.NUMBER_OF_CLASSES,
    #         "DATA_DISTRIBUTION": self.DATA_DISTRIBUTION,
    #         "ROUND_EPOCHS": self.NUMBER_OF_EPOCHS,
    #         "SENSITIVITY_PERCENTAGE": self.SENSITIVITY_PERCENTAGE,
    #         "DYNAMIC_SENSITIVITY_PERCENTAGE": self.DYNAMIC_SENSITIVITY_PERCENTAGE,
    #         "TRAIN_BATCH_SIZE": self.TRAIN_BATCH_SIZE,
    #         "TEST_BATCH_SIZE": self.TEST_BATCH_SIZE,
    #         "TRANSFORM_INPUT_SIZE": self.TRANSFORM_INPUT_SIZE,
    #         "LEARNING_RATE": 0.0001 if self.MODEL_TYPE == MODEL_VGG else 0.001,
    #         "WEIGHT_DECAY": self.WEIGHT_DECAY,
    #         "NUMBER_OF_CLIENTS": self.NUMBER_OF_CLIENTS,
    #         "DIRICHLET_BETA": self.DIRICHLET_BETA,
    #         "DESIRED_DISTRIBUTION": self.DESIRED_DISTRIBUTION,
    #         "SAVE_BEFORE_AGGREGATION_MODELS": self.SAVE_BEFORE_AGGREGATION_MODELS,
    #         "SAVE_GLOBAL_MODELS": self.SAVE_GLOBAL_MODELS,
    #         "DO_CLUSTER": self.DO_CLUSTER,
    #         "CLUSTERING_PERIOD": self.CLUSTERING_PERIOD,
    #         "FEDERATED_LEARNING_ROUNDS": self.FEDERATED_LEARNING_ROUNDS,
    #         "DISTANCE_METRIC": self.DISTANCE_METRIC,
    #         "GPU_INDEX": self.DISTANCE_METRIC,
    #         "DEVICE": self.DISTANCE_METRIC,
    #         "STOP_AVG_ACCURACY": self.DISTANCE_METRIC,
    #         "REMOVE_COMMON_IDS": self.REMOVE_COMMON_IDS,
    #         "FED_AVG": self.FED_AVG,
    #         "PRE_COMPUTED_DATA_DRIVEN_CLUSTERING": self.PRE_COMPUTED_DATA_DRIVEN_CLUSTERING,
    #         "DISTANCE_METRIC_ON_PARAMETERS": self.DISTANCE_METRIC_ON_PARAMETERS,
    #         self.PRETRAINED
    #     }
    #
    #     return config_dic

    def _validate_model_type(self, model_type: str) -> str:
        if model_type not in [
            MODEL_CNN,
            MODEL_RESNET_18,
            MODEL_RESNET_50,
            MODEL_MOBILENET,
            MODEL_VGG,
            MODEL_VIT,
            MODEL_SWIN,
        ]:
            raise TypeError(f"unsupported model type, {model_type}")

        return model_type

    def _validate_dataset_type(self, dataset_type: str) -> str:
        if dataset_type not in [
            DATA_SET_STL_10,
            DATA_SET_CIFAR_10,
            DATA_SET_CIFAR_100,
            DATA_SET_FMNIST,
            DATA_SET_SVHN,
            DATA_SET_TINY_IMAGE_NET
        ]:
            raise TypeError(f"unsupported dataset type, {dataset_type}")

        return dataset_type

    def _dataset_number_of_classes(self, dataset_type: str) -> int:
        if dataset_type == DATA_SET_CIFAR_100:
            return 100
        elif dataset_type == DATA_SET_TINY_IMAGE_NET:
            return 200
        else:
            return 10

    def _validate_data_distribution(self, data_distribution_kind: str, desired_distribution: str) -> str:
        if data_distribution_kind == DATA_DISTRIBUTION_FIX:
            if desired_distribution is None:
                raise TypeError(f"desired_distribution is None while the data_distribution_kind is fix")

            return "noniid-fix"

        elif data_distribution_kind == DATA_DISTRIBUTION_N_20:
            if self.DATASET_TYPE == DATA_SET_CIFAR_100:
                return "noniid-#label20"
            elif self.DATASET_TYPE == DATA_SET_TINY_IMAGE_NET:
                return "noniid-#label40"
            else:
                return "noniid-#label2"

        elif data_distribution_kind == DATA_DISTRIBUTION_N_30:
            if self.DATASET_TYPE == DATA_SET_CIFAR_100:
                return "noniid-#label30"
            elif self.DATASET_TYPE == DATA_SET_TINY_IMAGE_NET:
                return "noniid-#label60"
            else:
                return "noniid-#label3"
        elif data_distribution_kind == DATA_DISTRIBUTION_DIR:
            return "noniid-labeldir"
        else:
            raise TypeError(f"unsupported data distribution data distribution of, {data_distribution_kind}")

    def _set_distance_metric(self, metric: str) -> str:
        if metric not in [
            DISTANCE_COORDINATE,
            DISTANCE_COSINE,
            DISTANCE_EUCLIDEAN,
        ]:
            raise TypeError(f"unsupported metric type, {metric}")

        return metric

    def _set_number_of_epochs(self, number_of_epochs) -> int:
        if number_of_epochs is not None:
            return number_of_epochs

        if self.MODEL_TYPE in [MODEL_VGG, MODEL_RESNET_50, MODEL_VIT, MODEL_SWIN]:
            number_of_epochs = 10
        else:
            number_of_epochs = 1

        print(
            f"using default value for `NUMBER_OF_EPOCHS` which is {number_of_epochs} for model {self.MODEL_TYPE}"
        )

        return number_of_epochs

    def _set_sensitivity_percentage(self, sensitivity_percentage, dynamic_sensitivity_percentage) -> int:

        if dynamic_sensitivity_percentage:
            print(
                f"calculating the sensitivity percentage for model {self.MODEL_TYPE} dynamically"
            )
            return 100

        print(
            f"using default value for `SENSITIVITY_PERCENTAGE` which is {sensitivity_percentage}"
        )
        return sensitivity_percentage

    def _set_transformer(
            self,
            train_batch_size: int | None,  # Use modern union type hint (Python 3.10+) or Optional[int]
            test_batch_size: int | None,
            transform_input_size: int | None,
    ) -> Tuple[int, int, int]:  # Correct type hint syntax

        # If user provided all values, use them directly
        if (
                train_batch_size is not None
                and test_batch_size is not None
                and transform_input_size is not None
        ):
            return train_batch_size, test_batch_size, transform_input_size

        if self.MODEL_TYPE == MODEL_MOBILENET:
            default_train_batch = 64
            default_test_batch = 64
            default_input_size = 224
        elif self.MODEL_TYPE == MODEL_RESNET_50:
            default_train_batch = 64
            default_test_batch = 128
            default_input_size = 224
        elif self.MODEL_TYPE == MODEL_VIT:
            default_train_batch = 32
            default_test_batch = 64
            default_input_size = 224
        elif self.MODEL_TYPE == MODEL_SWIN:
            default_train_batch = 32
            default_test_batch = 64
            default_input_size = 224
        else:
            print(f"Unknown MODEL_TYPE '{self.MODEL_TYPE}'. Using generic defaults.")
            default_train_batch = 32
            default_test_batch = 64
            default_input_size = 224

        final_train_batch_size = train_batch_size if train_batch_size is not None else default_train_batch
        final_test_batch_size = test_batch_size if test_batch_size is not None else default_test_batch
        final_transform_input_size = transform_input_size if transform_input_size is not None else default_input_size

        if train_batch_size is None:
            print(
                f"Using default value for `TRAIN_BATCH_SIZE` ({final_train_batch_size}) for model type {self.MODEL_TYPE}"
            )
        if test_batch_size is None:
            print(
                f"Using default value for `TEST_BATCH_SIZE` ({final_test_batch_size}) for model type {self.MODEL_TYPE}"
            )
        if transform_input_size is None:
            print(
                f"Using default value for `TRANSFORM_INPUT_SIZE` ({final_transform_input_size}) for model type {self.MODEL_TYPE}"
            )

        return (
            final_train_batch_size,
            final_test_batch_size,
            final_transform_input_size,
        )

    def _device(self, device, gpu_index) -> str:

        if gpu_index is not None:
            device = torch.device(f"cuda:{gpu_index}")

        gpus = list_available_gpus()

        if gpus:
            print("Available GPUs:")
            if device.type == 'cuda' and device.index is None:
                allocated_index = 0
            else:
                allocated_index = device.index

            for index, name in gpus:
                if device.type == 'cuda' and index == allocated_index:
                    print(f"Index: {index}, Device: {name} (ALLOCATED)")
                else:
                    print(f"Index: {index}, Device: {name} (UNALLOCATED)")
        else:
            raise Exception(f"given device is {device} while there is no gpu available!")

        return device

    def _stop_avg_accuracy(self, stop_avg_accuracy):
        if stop_avg_accuracy is None:
            return 0.1
        return stop_avg_accuracy
