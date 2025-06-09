import torch
from typing import Tuple, List

from torch.utils.data import DataLoader

from constants.client_roles_constants import (
    TRAIN, 
    TEST, 
    EVAL, 
    TRAIN_TEST, 
    TRAIN_EVAL, 
    TEST_EVAL, 
    TRAIN_TEST_EVAL
    )

from constants.loss_constants import (
    LOSS_MASKED_CROSS_ENTROPY, 
    LOSS_CROSS_ENTROPY, 
    LOSS_SMOOTHED_CROSS_ENTROPY
    )
from validators.validator_exporters import (
    transformer_model_size_exporter, 
    optimizer_constans_exporter
)
from constants.distances_constants import (
    DISTANCE_COORDINATE,
    DISTANCE_COSINE,
    DISTANCE_EUCLIDEAN,
)
from constants.schema_constants import (
    TRADITIONAL_FEDERATED_LEARNING, 
    CLUSTER_FEDERATED_LEARNING,
    DECENTRALIZED_FEDERATED_LEARNING
    )

from constants.topology_constants import  (
    TOPOLOGY_STAR, 
    TOPOLOGY_RING, 
    TOPOLOGY_K_CONNECT, 
    TOPOLOGY_CUSTOM
)
from constants.models_constants import (
    MODEL_CNN,
    MODEL_RESNET_18,
    MODEL_RESNET_50,
    MODEL_MOBILENET,
    MODEL_VGG,
    MODEL_VIT,
    MODEL_SWIN, 
    MODEL_LENET,
    MODEL_BERT,
    MODEL_ALBERT,
    TRANSFORMER_MODEL_SIZE_LARGE,
    TRANSFORMER_MODEL_SIZE_BASE,
)

from constants.aggregation_strategy_constants import (
    AGGREGATION_STRATEGY_FED_AVG,
    AGGREGATION_STRATEGY_FED_PROX,
)

from constants.datasets_constants import (
    DATA_SET_MNIST,
    DATA_SET_FMNIST,
    DATA_SET_FEMNIST,
    DATA_SET_CIFAR_10,
    DATA_SET_CIFAR_100,
    DATA_SET_SVHN,
    DATA_SET_STL_10,
    DATA_SET_TINY_IMAGE_NET,
    DATA_SET_SHAKESPEARE,
    DATA_SET_BBC,
    DATA_SET_YAHOO,
    DATA_SET_NEWS,
    SHAKESPEARE_RAW_DATA_PATH,
    SHAKESPEARE_TARGET_PATH,
)

from constants.data_distribution_constants import (
    DATA_DISTRIBUTION_FIX,
    DATA_DISTRIBUTION_IID,
    DATA_DISTRIBUTION_IID_DIFF_QUANTITY,
    DATA_DISTRIBUTION_N_10,
    DATA_DISTRIBUTION_N_20,
    DATA_DISTRIBUTION_N_30,
    DATA_DISTRIBUTION_N_40,
    DATA_DISTRIBUTION_N_50,
    DATA_DISTRIBUTION_N_60,
    DATA_DISTRIBUTION_N_70,
    DATA_DISTRIBUTION_N_80,
    DATA_DISTRIBUTION_N_90,
    DATA_DISTRIBUTION_DIR,
    DATA_DISTRIBUTION_REAL_FEMNIST
)

from utils.gpu_index_list import list_available_gpus
from utils.log import Log
from validators.runtime_config import RuntimeConfig


class ConfigValidator:
    def __init__(
            self,
            learning_rate: float,
            model_type: str,
            transformer_model_size: str,
            dataset_type: str,
            loss_function: str,
            optimizer: str,
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
            random_seed: int = 42,
            fed_avg: bool = False,
            chunking_with_gradients: bool = False,
            chunking_parts: float = 5.0,
            chunking_random_section: bool = False,
            stop_avg_accuracy=None,
            pre_computed_data_driven_clustering: bool = False,
            distance_metric_on_parameters: bool = True,
            pretrained_models: bool = False,
            federated_learning_schema: str = None,
            federated_learning_topology: str = None,
            client_k_neighbors: int = None,
            client_role: str = None,
            client_sampling_rate: float = 1.0,
            mean_accuracy_to_csv: bool = True,
            federation_id: str = "",
            encryption_method: str = None,
            xmkckks_weight_decimals: int = None,
            use_global_accuracy_for_noniid: bool = True,

    ):

        self._RUNTIME_COMFIG: RuntimeConfig | None = None

        self.RANDOM_SEED = random_seed

        self.LEARNING_RATE = learning_rate
        self.MODEL_TYPE = self._validate_model_type(model_type)
        self.TRANSFORMER_MODEL_SIZE = self._transformer_model_size(transformer_model_size)
        self.DATASET_TYPE = self._validate_dataset_type(dataset_type)
        self.LOSS_FUNCTION = self._validate_loss_function(loss_function)
        self.OPTIMIZER = self._validate_optimizer(optimizer)
        self.DATA_DISTRIBUTION = self._validate_data_distribution(data_distribution_kind, desired_distribution)
        self.DISTANCE_METRIC = self._set_distance_metric(distance_metric)
        self.NUMBER_OF_EPOCHS = self._set_number_of_epochs(number_of_epochs)
        self.DYNAMIC_SENSITIVITY_PERCENTAGE = dynamic_sensitivity_percentage
        self.SENSITIVITY_PERCENTAGE = self._set_sensitivity_percentage(sensitivity_percentage, dynamic_sensitivity_percentage)

        self.TRAIN_BATCH_SIZE, self.TEST_BATCH_SIZE, self.TRANSFORM_INPUT_SIZE = (
            self._set_transformer(
                train_batch_size,
                test_batch_size,
                transform_input_size,
            )
        )

        self.WEIGHT_DECAY = weight_decay
        self.NUMBER_OF_CLIENTS = number_of_clients
        self.NUMBER_OF_CLASSES = self._dataset_number_of_classes(self.DATASET_TYPE)
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
        self.MULTI_GPU = self._is_multi_gpu(gpu_index)
        self.GPU_DEVICE_IDS = self._parse_gpu_indices(gpu_index)
        self.FED_AVG = fed_avg
        self.CHUNKING_WITH_GRADIENTS = chunking_with_gradients
        self.CHUNKING_PARTS = chunking_parts
        self.CHUNKING_RANDOM_SECTION = chunking_random_section
        self.CHUNKING = self._validata_chunking(chunking)
        self.STOP_AVG_ACCURACY = self._stop_avg_accuracy(stop_avg_accuracy)
        self.PRE_COMPUTED_DATA_DRIVEN_CLUSTERING = pre_computed_data_driven_clustering
        self.DISTANCE_METRIC_ON_PARAMETERS = distance_metric_on_parameters
        self.PRETRAINED_MODELS = pretrained_models
        self.FEDERATED_LEARNING_SCHEMA = self._federated_learning_schema(federated_learning_schema)
        self.FEDERATED_LEARNING_TOPOLOGY = self._schem_n_toplogy_macher(federated_learning_schema, federated_learning_topology)
        self.CLIENT_K_NEIGHBORS = self._validate_client_k_neighbors(number_of_clients, federated_learning_topology, client_k_neighbors)
        self.CLIENT_ROLE = self._client_role(client_role)
        self.CLIENT_SAMPLING_RATE = client_sampling_rate
        self.AGGREGATION_STRATEGY = self._aggregation_strategy(aggregation_strategy)
        self.AGGREGATION_SAMPLE_SCALING = aggregation_sample_scaling
        self.FEDERATION_ID = federation_id
        self.MEAN_ACCURACY_TO_CSV = mean_accuracy_to_csv
        self.USE_GLOBAL_ACCURACY_FOR_NONIID = use_global_accuracy_for_noniid

    @property
    def RUNTIME_COMFIG(self) -> RuntimeConfig:
        return self._RUNTIME_COMFIG

    @RUNTIME_COMFIG.setter
    def RUNTIME_COMFIG(self, runtime_config: RuntimeConfig ) -> None:
        self._RUNTIME_COMFIG: RuntimeConfig = runtime_config

    def _validate_model_type(self, model_type: str) -> str:
        if model_type not in [
            MODEL_CNN,
            MODEL_RESNET_18,
            MODEL_RESNET_50,
            MODEL_MOBILENET,
            MODEL_VGG,
            MODEL_VIT,
            MODEL_SWIN, 
            MODEL_LENET,
            MODEL_BERT,
            MODEL_ALBERT,
            TRANSFORMER_MODEL_SIZE_LARGE,
            TRANSFORMER_MODEL_SIZE_BASE,
        ]:
            raise TypeError(f"unsupported model type, {model_type}")

        return model_type
    
    def _transformer_model_size(self, transformer_model_size):
        if transformer_model_size in transformer_model_size_exporter():
            return transformer_model_size
        else :
            raise TypeError (f"{transformer_model_size} is not valid transformer model size")


    def _validate_dataset_type(self, dataset_type: str) -> str:
        if dataset_type not in [
            DATA_SET_MNIST,
            DATA_SET_FMNIST,
            DATA_SET_FEMNIST,
            DATA_SET_CIFAR_10,
            DATA_SET_CIFAR_100,
            DATA_SET_SVHN,
            DATA_SET_STL_10,
            DATA_SET_TINY_IMAGE_NET,
            DATA_SET_SHAKESPEARE,
            DATA_SET_BBC,
            DATA_SET_YAHOO,
            DATA_SET_NEWS,
            SHAKESPEARE_RAW_DATA_PATH,
            SHAKESPEARE_TARGET_PATH,
        ]:
            raise TypeError(f"unsupported dataset type, {dataset_type}")

        return dataset_type

    def _validate_loss_function(self, loss_function: str) -> str:
        if loss_function not in [
            LOSS_CROSS_ENTROPY,
            LOSS_MASKED_CROSS_ENTROPY,
            LOSS_SMOOTHED_CROSS_ENTROPY,
        ]:
            raise TypeError(f"unsupported loss_function, {loss_function}")

        return loss_function

    def _validate_optimizer(self, optimizer):
        if optimizer not in optimizer_constans_exporter():
            raise TypeError(f"unsupported optimizer, {optimizer}")
        return optimizer
    
    def _dataset_number_of_classes(self, dataset_type: str) -> int:
        if dataset_type == DATA_SET_CIFAR_100:
            return 100
        elif dataset_type == DATA_SET_TINY_IMAGE_NET:
            return 200
        elif dataset_type == DATA_SET_BBC:
            return 5
        else:
            return 10

    def _validate_data_distribution(
        self, data_distribution_kind: str, desired_distribution: str
    ) -> str:
        if data_distribution_kind == DATA_DISTRIBUTION_FIX:
            if desired_distribution is None:
                raise TypeError(
                    f"desired_distribution is None while the data_distribution_kind is fix"
                )

            return "noniid-fix"

        elif data_distribution_kind == DATA_DISTRIBUTION_N_10:
            if self.DATASET_TYPE == DATA_SET_CIFAR_100:
                return "noniid-#label10"
            elif self.DATASET_TYPE == DATA_SET_TINY_IMAGE_NET:
                return "noniid-#label20"
            else:
                return "noniid-#label1"

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

        elif data_distribution_kind == DATA_DISTRIBUTION_N_40:
            if self.DATASET_TYPE == DATA_SET_CIFAR_100:
                return "noniid-#label40"
            elif self.DATASET_TYPE == DATA_SET_TINY_IMAGE_NET:
                return "noniid-#label80"
            else:
                return "noniid-#label4"

        elif data_distribution_kind == DATA_DISTRIBUTION_N_50:
            if self.DATASET_TYPE == DATA_SET_CIFAR_100:
                return "noniid-#label50"
            elif self.DATASET_TYPE == DATA_SET_TINY_IMAGE_NET:
                return "noniid-#label100"
            else:
                return "noniid-#label5"

        elif data_distribution_kind == DATA_DISTRIBUTION_N_60:
            if self.DATASET_TYPE == DATA_SET_CIFAR_100:
                return "noniid-#label60"
            elif self.DATASET_TYPE == DATA_SET_TINY_IMAGE_NET:
                return "noniid-#label120"
            else:
                return "noniid-#label6"

        elif data_distribution_kind == DATA_DISTRIBUTION_N_70:
            if self.DATASET_TYPE == DATA_SET_CIFAR_100:
                return "noniid-#label70"
            elif self.DATASET_TYPE == DATA_SET_TINY_IMAGE_NET:
                return "noniid-#label140"
            else:
                return "noniid-#label7"

        elif data_distribution_kind == DATA_DISTRIBUTION_N_80:
            if self.DATASET_TYPE == DATA_SET_CIFAR_100:
                return "noniid-#label80"
            elif self.DATASET_TYPE == DATA_SET_TINY_IMAGE_NET:
                return "noniid-#label160"
            else:
                return "noniid-#label8"

        elif data_distribution_kind == DATA_DISTRIBUTION_N_90:
            if self.DATASET_TYPE == DATA_SET_CIFAR_100:
                return "noniid-#label90"
            elif self.DATASET_TYPE == DATA_SET_TINY_IMAGE_NET:
                return "noniid-#label180"
            else:
                return "noniid-#label9"

        elif data_distribution_kind == DATA_DISTRIBUTION_DIR:
            return "noniid-labeldir"
        elif data_distribution_kind == DATA_DISTRIBUTION_IID:
            return "homo"
        elif data_distribution_kind == DATA_DISTRIBUTION_IID_DIFF_QUANTITY:
            return "iid-diff-quantity"
        elif data_distribution_kind == DATA_DISTRIBUTION_REAL_FEMNIST:
            if self.DATASET_TYPE != DATA_SET_FEMNIST:
                raise TypeError(f"expected femnist dataset but got: {data_distribution_kind}")
            return "real"
        else:
            raise TypeError(
                f"unsupported data distribution data distribution of, {data_distribution_kind}"
            )

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
            train_batch_size: int | None,  
            test_batch_size: int | None,
            transform_input_size: int | None,
    ) -> Tuple[int, int, int]:

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
            print(f"MODEL_TYPE is '{self.MODEL_TYPE}'. Using generic defaults for:")
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

        if device == "cpu":
            return device

        available_gpus = list_available_gpus()
        
        if not available_gpus:
            raise Exception(
                f"Given device is {device} while there is no GPU available!"
            )

        if self._is_multi_gpu(gpu_index):
            gpu_indices = self._parse_gpu_indices(gpu_index)
            print(f"Multi-GPU setup requested with GPUs: {gpu_indices}")
            
            print("Available GPUs:")
            for index, name in available_gpus:
                if index in gpu_indices:
                    print(f"Index: {index}, Device: {name} (ALLOCATED)")
                else:
                    print(f"Index: {index}, Device: {name} (UNALLOCATED)")
            
            primary_device = f"cuda:{gpu_indices[0]}"
            return primary_device

        if gpu_index is not None:
            single_gpu_indices = self._parse_gpu_indices(gpu_index)
            if single_gpu_indices:
                allocated_index = single_gpu_indices[0]
                device = torch.device(f"cuda:{allocated_index}")
            else:
                raise Exception(f"GPU index {gpu_index} is not available!")
        else:
            allocated_index = available_gpus[0][0]
            device = torch.device(f"cuda:{allocated_index}")

        print("Available GPUs:")
        for index, name in available_gpus:
            if index == allocated_index:
                print(f"Index: {index}, Device: {name} (ALLOCATED)")
            else:
                print(f"Index: {index}, Device: {name} (UNALLOCATED)")

        return str(device)

    def _stop_avg_accuracy(self, stop_avg_accuracy):
        if stop_avg_accuracy is None:
            return 0.1
        return stop_avg_accuracy

    def _federated_learning_schema(self, federated_learning_schema: str):
        if federated_learning_schema not in [
            TRADITIONAL_FEDERATED_LEARNING,
            CLUSTER_FEDERATED_LEARNING,
            DECENTRALIZED_FEDERATED_LEARNING,
        ]:
            raise TypeError(f"unknown federated_learning_schema type: {federated_learning_schema}")
        return federated_learning_schema

    def _schem_n_toplogy_macher(self, federated_learning_schema: str, federated_learning_topology: str):
        if federated_learning_schema == TRADITIONAL_FEDERATED_LEARNING and federated_learning_topology != TOPOLOGY_STAR:
            raise TypeError(f"Traditional federated learning uses the star topology only!")
        return federated_learning_topology


    def _client_role(self, client_role: str):
        if client_role not in [
            TRAIN,
            TEST,
            EVAL,
            TRAIN_TEST,
            TRAIN_EVAL,
            TEST_EVAL,
            TRAIN_TEST_EVAL,
        ]:
            raise TypeError(f"unknown client_role type: {client_role}")
        return client_role


    def _aggregation_strategy(self, aggregation_strategy: str | None) -> str:
        if aggregation_strategy not in [
            AGGREGATION_STRATEGY_FED_AVG,
            AGGREGATION_STRATEGY_FED_PROX,
        ]:
            raise TypeError(f"unknown aggregation_strategy type: {aggregation_strategy}")
        return aggregation_strategy
    
    def _validate_client_k_neighbors(self, number_of_clients: int, federated_learning_topology: str, client_k_neighbors: int):
        if federated_learning_topology == 'k_connect' and client_k_neighbors is not None:
            if number_of_clients > client_k_neighbors > 1: 
                return client_k_neighbors
            else: 
                raise TypeError(f"number_of_clients must be greater than client_k_neighbors and client_k_neighbors must be greater than 1"
                                f" while the federated_learning_topology is k_connect")
        elif federated_learning_topology == 'k_connect' and client_k_neighbors is None:
            raise TypeError(f"client_k_neighbors is None while the federated_learning_topology is k_connect")
        else:
            return None
        
    def _validata_chunking(self, chunking: bool) -> bool:
        if chunking and self.SENSITIVITY_PERCENTAGE == 100 and self.DYNAMIC_SENSITIVITY_PERCENTAGE:
            warnings.warn(f"got value sensitivity_percentage: {self.SENSITIVITY_PERCENTAGE} and dynamic_sensitivity_percentage: {self.DYNAMIC_SENSITIVITY_PERCENTAGE}"
                          f"which results in calculating optimal pruning rate!")
        
        if chunking and not self.CHUNKING_WITH_GRADIENTS:
            import sys
            print("=" * 80, file=sys.stderr)
            print("CRITICAL CONFIGURATION ERROR: CHUNKING WITHOUT GRADIENT ANALYSIS!", file=sys.stderr)
            print("When chunking=True, chunking_with_gradients MUST be True.", file=sys.stderr)
            print("Otherwise, chunk selection will be random instead of importance-based!", file=sys.stderr)
            print("Solution: Set 'chunking_with_gradients: true' in your configuration.", file=sys.stderr)
            print("=" * 80, file=sys.stderr)
            warnings.warn("CHUNKING ENABLED WITHOUT GRADIENT ANALYSIS - THIS WILL USE RANDOM CHUNK SELECTION!")
            
        return chunking

    def _is_multi_gpu(self, gpu_index):
        """Check if multi-GPU is requested based on gpu_index format"""
        if gpu_index is None:
            return False
        
        if isinstance(gpu_index, str) and ":" in gpu_index:
            return True
        
        return False
    
    def _parse_gpu_indices(self, gpu_index):
        """Parse GPU indices from gpu_index parameter"""
        if gpu_index is None:
            return []
        
        available_gpus = list_available_gpus()
        available_indices = [index for index, _ in available_gpus]
        
        if isinstance(gpu_index, str) and ":" in gpu_index:
            try:
                start, end = map(int, gpu_index.split(":"))
                requested_indices = list(range(start, end))
                valid_indices = [idx for idx in requested_indices if idx in available_indices]
                return valid_indices
            except ValueError:
                raise ValueError(f"Invalid GPU range format: {gpu_index}. Expected format: 'start:end'")
        
        elif isinstance(gpu_index, (int, str)):
            try:
                single_index = int(gpu_index)
                if single_index in available_indices:
                    return [single_index]
                else:
                    raise ValueError(f"GPU index {single_index} not available. Available indices: {available_indices}")
            except ValueError:
                raise ValueError(f"Invalid GPU index: {gpu_index}")
        
        return []