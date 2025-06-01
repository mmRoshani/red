import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split

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

def load_and_tokenize(data_dir, config, pretrained_model='bert-base-uncased', max_length=512, pad_to_max=True, test_size=0.2, shuffle=True):
# (#
#     data_dir,
#     config,
#     pretrained_model='bert-base-uncased',
#     max_length=512, # todo I thin max_lenght is defined in the config
#     pad_to_max=True,
#     test_size=0.2,
#     shuffle=True
# ):
    # 1. Gather texts and labels
    train_loaders = []
    test_loaders = []

    tokenizer = BertTokenizer.from_pretrained(pretrained_model)
    data_dir += "/bbc"
    # Iterate over client directories
    for client_name in sorted(os.listdir(data_dir)):
        client_path = os.path.join(data_dir, client_name)
        if not os.path.isdir(client_path):
            continue

        print(f"Processing {client_name}...")

        texts = []
        labels = []
        label2id = {}

        # Read each label directory in the client folder
        for label_id, label_name in enumerate(sorted(os.listdir(client_path))):
            label_path = os.path.join(client_path, label_name)
            if not os.path.isdir(label_path):
                continue

            label2id[label_name] = label_id

            for fn in os.listdir(label_path):
                if fn.lower().endswith('.txt'):
                    path = os.path.join(label_path, fn)
                    raw = robust_read(path)
                    texts.append(raw.strip())
                    labels.append(label_id)

        if not texts:  # If client has no datasets, skip
            continue

        # Tokenization
        encodings = tokenizer(
            texts,
            max_length=max_length,
            padding='max_length' if pad_to_max else True,
            truncation=True,
            return_tensors='pt'
        )

        # Train/test split
        train_idx, test_idx = train_test_split(
            list(range(len(labels))),
            test_size=test_size,
            shuffle=shuffle,
            stratify=labels
        )

        train_encodings = {key: val[train_idx] for key, val in encodings.items()}
        test_encodings = {key: val[test_idx] for key, val in encodings.items()}
        train_labels = [labels[i] for i in train_idx]
        test_labels = [labels[i] for i in test_idx]

        train_dataset = TextDataset(train_encodings, train_labels)
        test_dataset = TextDataset(test_encodings, test_labels)

        train_loader = DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=config.TEST_BATCH_SIZE, shuffle=True)

        train_loaders.append(train_loader)
        test_loaders.append(test_loader)

    return train_loaders, test_loaders


