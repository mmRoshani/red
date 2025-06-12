import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split

from src.utils.log import Log
from src.validators.config_validator import ConfigValidator


# Dummy placeholder for robust_read and TextDataset
def robust_read(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

 # TODO: I think max_lenght is defined in the config
def load_and_tokenize(
        data_dir: str,
        config: 'ConfigValidator',
        log: 'Log',
        pretrained_model: str = 'bert-base-uncased',
        max_length: int = 512,
        pad_to_max: bool = True,
        test_size: float = 0.2,  # ignored now
        shuffle: bool = True     # used only in dataloader
):
    train_loaders = []
    test_loaders = []


    tokenizer = BertTokenizer.from_pretrained(pretrained_model)

    # Normalize paths
    data_dir = os.path.abspath(os.path.expanduser(data_dir))
    test_dir = os.path.abspath(os.path.expanduser(data_dir.replace("bbc_train", "bbc_test")))

    if not data_dir.rstrip('/').endswith('bbc_train'):
        data_dir += "/bbc_train"
    if not test_dir.rstrip('/').endswith('bbc_test'):
        test_dir += "/bbc_test"

    for client_id in sorted(os.listdir(data_dir)):
        train_client_path = os.path.join(data_dir, client_id)
        test_client_path = os.path.join(test_dir, client_id)

        if not os.path.isdir(train_client_path) or not os.path.isdir(test_client_path):
            continue

        log.info(f'Preprocessing client no #: {client_id}')

        # -------- TRAIN DATA --------
        train_texts, train_labels = [], []
        label2id = {}

        for label_id, label_name in enumerate(sorted(os.listdir(train_client_path))):
            label_path = os.path.join(train_client_path, label_name)
            if not os.path.isdir(label_path):
                continue

            label2id[label_name] = label_id

            for fn in os.listdir(label_path):
                if fn.lower().endswith('.txt'):
                    path = os.path.join(label_path, fn)
                    raw = robust_read(path)
                    train_texts.append(raw.strip())
                    train_labels.append(label_id)

        if not train_texts:
            continue

        train_encodings = tokenizer(
            train_texts,
            max_length=max_length,
            padding='max_length' if pad_to_max else True,
            truncation=True,
            return_tensors='pt'
        )
        train_dataset = TextDataset(train_encodings, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=shuffle)
        train_loaders.append(train_loader)

        # -------- TEST DATA --------
        test_texts, test_labels = [], []

        for label_name in sorted(label2id.keys()):
            label_id = label2id[label_name]
            label_path = os.path.join(test_client_path, label_name)
            if not os.path.isdir(label_path):
                continue

            for fn in os.listdir(label_path):
                if fn.lower().endswith('.txt'):
                    path = os.path.join(label_path, fn)
                    raw = robust_read(path)
                    test_texts.append(raw.strip())
                    test_labels.append(label_id)

        if not test_texts:
            continue

        test_encodings = tokenizer(
            test_texts,
            max_length=max_length,
            padding='max_length' if pad_to_max else True,
            truncation=True,
            return_tensors='pt'
        )
        test_dataset = TextDataset(test_encodings, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=config.TEST_BATCH_SIZE, shuffle=shuffle)
        test_loaders.append(test_loader)

    return train_loaders, test_loaders



