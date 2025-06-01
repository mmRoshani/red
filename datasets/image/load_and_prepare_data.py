from datasets.image.get_dataloader import get_dataloader
from datasets.image.partition_data import partition_data
from validators.config_validator import ConfigValidator
from utils.log import Log


def load_and_prepare_data(config: "ConfigValidator", log: "Log"):
    (
        X_train,
        y_train,
        X_test,
        y_test,
        net_dataidx_map,
        # test_dataidx_map,
        traindata_cls_counts,
    ) = partition_data(
        dataset=config.DATASET_TYPE,
        datadir="~/datasets/",
        logdir="./logs/",
        partition=config.DATA_DISTRIBUTION,
        n_parties=config.NUMBER_OF_CLIENTS,
        log=log,
        beta=config.DIRICHLET_BETA,
    )
    train_loaders = []
    test_loaders = []
    for client_id in range(config.NUMBER_OF_CLIENTS):
        dataidxs = net_dataidx_map[client_id]
        # testidxs = test_dataidx_map[client_id]

        train_dl_local, test_dl_local, _, _ = get_dataloader(
            dataset=config.DATASET_TYPE,
            datadir="~/datasets/",
            train_bs=config.TRAIN_BATCH_SIZE,
            test_bs=config.TEST_BATCH_SIZE,
            dataidxs=dataidxs,
            # testidxs=testidxs, # TODO: fix the test datasets distribution
        )
        train_loaders.append(train_dl_local)
        test_loaders.append(test_dl_local)

    return train_loaders, test_loaders
