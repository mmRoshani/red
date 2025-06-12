<<<<<<< HEAD:src/datasets/text/shakespeare/shakespear_factory.py
import osSHAKESPEARE_RAW_DATA_PATH = "datasets/by_play_and_character"
SHAKESPEARE_TARGET_PATH = "shakespear_data/"
import re
import time
import random
from constants.datasets_constants import SHAKESPEARE_RAW_DATA_PATH, SHAKESPEARE_TARGET_PATH


def train_test_split(text_path, frac):
    r"""
    splits role text datasets into a set of training lines (the first `frac` of lines for the role),
     and test lines (the last 1 - `frac`, rounded up to at least one line)
    :param text_path: path to text file
    :param frac: training fraction
    return `train_text`, `test_text`
    """
    assert 0 < frac < 1, "`frac` should be in (0, 1)"

    with open(text_path, "r") as f:
        raw_text = f.read()

    raw_text = re.sub(r"   *", r' ', raw_text)

    all_lines = raw_text.split('\n')[:-1]
    n_lines = len(all_lines)

    n_test_lines = max(1, int((1-frac)*n_lines))
    n_train_lines = n_lines - n_test_lines

    train_lines = all_lines[:n_train_lines]
    test_lines = all_lines[n_train_lines:]

    train_text = ' '.join(train_lines)
    test_text = ' '.join(test_lines)

    return train_text, test_text


def save_task(dir_path, train_text, test_text):
    r"""
    save `train_text` and `test_text` as `.txt` files in `dir_path`
    :param train_text:
    :param test_text:
    :param dir_path:
    """
    with open(os.path.join(dir_path, "train.txt"), 'w') as f:
        f.write(train_text)

    with open(os.path.join(dir_path, "test.txt"), 'w') as f:
        f.write(test_text)


def shakespear_factory(dataset_usage_fraction: float = 1.0,
                       train_fraction: float = 0.8,
                       train_fraction_per_client: float = 1.0, # fraction of tasks / clients  participating to the training; default is 1.0
                        seed: int = 42,
                        ):

    rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
    rng = random.Random(rng_seed)

    n_tasks = int(len(os.listdir(SHAKESPEARE_RAW_DATA_PATH)) * dataset_usage_fraction)
    file_names_list = os.listdir(SHAKESPEARE_RAW_DATA_PATH)
    rng.shuffle(file_names_list)

    file_names_list = file_names_list[:n_tasks]
    rng.shuffle(file_names_list)

    os.makedirs(os.path.join(SHAKESPEARE_TARGET_PATH, "train"), exist_ok=True)
    os.makedirs(os.path.join(SHAKESPEARE_TARGET_PATH, "test"), exist_ok=True)

    for idx, file_name in enumerate(file_names_list):
        if idx < int(train_fraction_per_client * n_tasks):
            mode = "train"
        else:
            mode = "test"

        text_path = os.path.join(SHAKESPEARE_RAW_DATA_PATH, file_name)
        train_text, test_text = train_test_split(text_path, train_fraction)

        save_path = os.path.join(SHAKESPEARE_TARGET_PATH, mode, f"task_{idx}")
        os.makedirs(save_path, exist_ok=True)

        save_task(
            dir_path=save_path,
            train_text=train_text,
            test_text=test_text
        )



=======
import os
import re
import time
import random
from constants.datasets_constants import SHAKESPEARE_RAW_DATA_PATH, SHAKESPEARE_TARGET_PATH
from validators.config_validator import ConfigValidator


def train_test_split(text_path, frac):
    r"""
    splits role text data into a set of training lines (the first `frac` of lines for the role),
    and test lines (the last 1 - `frac`, rounded up to at least one line)
    :param text_path: path to text file
    :param frac: training fraction
    :param n_clients: number of clients  #todo At first, code checks if the n_clients is defiend to use as number of cliants. Else it uses frac to compute the number of classes.
    return `train_text`, `test_text`
    """
    assert 0 < frac < 1, "`frac` should be in (0, 1)"

    with open(text_path, "r") as f:
        raw_text = f.read()

    raw_text = re.sub(r"   *", r' ', raw_text)

    all_lines = raw_text.split('\n')[:-1]
    n_lines = len(all_lines)

    n_test_lines = max(1, int((1-frac)*n_lines))
    n_train_lines = n_lines - n_test_lines

    train_lines = all_lines[:n_train_lines]
    test_lines = all_lines[n_train_lines:]

    train_text = ' '.join(train_lines)
    test_text = ' '.join(test_lines)

    return train_text, test_text


def save_task(dir_path, train_text, test_text):
    r"""
    save `train_text` and `test_text` as `.txt` files in `dir_path`
    :param train_text:
    :param test_text:
    :param dir_path:
    """
    with open(os.path.join(dir_path, "train.txt"), 'w') as f:
        f.write(train_text)

    with open(os.path.join(dir_path, "test.txt"), 'w') as f:
        f.write(test_text)


def shakespear_factory(dataset_usage_fraction: float = 1.0,
                        n_clients: int = None,
                        train_fraction: float = 0.8,
                        train_fraction_per_client: float = 1.0, # fraction of tasks / clients  participating to the training; default is 1.0
                        seed: int = 42,
                        ):

    rng_seed = (seed if (seed is not None and seed >= 0) else int(time.time()))
    rng = random.Random(rng_seed)


    all_files = os.listdir(SHAKESPEARE_RAW_DATA_PATH)
    total_roles = len(all_files)

    if n_clients is not None:
        n_clients = min(n_clients, total_roles)
    else:
        n_clients = int(total_roles * dataset_usage_fraction)

    file_names_list = os.listdir(SHAKESPEARE_RAW_DATA_PATH)
    rng.shuffle(file_names_list)

    file_names_list = file_names_list[:n_clients]
    rng.shuffle(file_names_list)

    os.makedirs(os.path.join(SHAKESPEARE_TARGET_PATH, "train"), exist_ok=True)
    os.makedirs(os.path.join(SHAKESPEARE_TARGET_PATH, "test"), exist_ok=True)

    for idx, file_name in enumerate(file_names_list):
        if idx < int(train_fraction_per_client * n_clients):
            mode = "train"
        else:
            mode = "test"

        text_path = os.path.join(SHAKESPEARE_RAW_DATA_PATH, file_name)
        train_text, test_text = train_test_split(text_path, train_fraction)

        save_path = os.path.join(SHAKESPEARE_TARGET_PATH, mode, f"task_{idx}")
        os.makedirs(save_path, exist_ok=True)

        save_task(
            dir_path=save_path,
            train_text=train_text,
            test_text=test_text
        )




def dl_generator_shakespeare(config: 'ConfigValidator', dataset: str, data_dir: str, batch_size: int, chunk_len, is_validation=False):
    shakespear_factory(dataset_usage_fraction=0.01, n_clients=config.NUMBER_OF_CLIENTS, train_fraction=0.8, train_fraction_per_client=1.0, seed=42)
    inputs, targets = None, None
    train_iterators, val_iterators, test_iterators = [], [], []

    for task_id, task_dir in enumerate(os.listdir(data_dir)):
        task_data_path = os.path.join(data_dir, task_dir)

        train_iterator = get_spe_loader(
        dataset=dataset,
        path=os.path.join(task_data_path, f"train{dataset}"),
        batch_size=config.TRAIN_BATCH_SIZE, chunk_len=chunk_len, inputs=inputs, targets=targets, train=True
        )

        val_iterator = get_spe_loader(
        dataset=dataset,
        path=os.path.join(task_data_path, f"train{dataset}"),
        batch_size=config.TEST_BATCH_SIZE, chunk_len=chunk_len, inputs=inputs, targets=targets, train=False
        )

        if is_validation:
            test_set = "val"
        else:
            test_set = "test"

        test_iterator =get_spe_loader(
        dataset=dataset,
        path=os.path.join(task_data_path, f"{test_set}{dataset}"),
        batch_size=batch_size, chunk_len=chunk_len, inputs=inputs, targets=targets, train=False
        )

        if test_iterator!=None:
            train_iterators.append(train_iterator)
            val_iterators.append(val_iterator)
            test_iterators.append(test_iterator)

    # original_client_num = task_id + 1

    # return train_iterators, val_iterators, test_iterators, original_client_num
    return train_iterators, test_iterators


def get_spe_loader(dataset, path, batch_size, train, chunk_len=5, inputs=None, targets=None):


    # This dataset is designed for next-character prediction (language modeling),
    # where the goal is to predict the next character in a text sequence,
    # not for classification tasks where the output would be a fixed class label.
    dataset = CharacterDataset(path, chunk_len=chunk_len)
    # the CharacterDataset is defined in the FedTP/datasets.py so we can use it here.

    if len(dataset) == 0:
        return

    drop_last = (len(dataset) > batch_size) and train

    return data.DataLoader(dataset, batch_size=batch_size, shuffle=train, drop_last=drop_last)


>>>>>>> dev:datasets/text/shakespeare/shakespear_factory.py
