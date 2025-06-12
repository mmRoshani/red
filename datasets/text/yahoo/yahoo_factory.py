import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

from constants.data_distribution_constants import DATA_DISTRIBUTION_STANDARD_NON_IID_DIR, \
    DATA_DISTRIBUTION_STANDARD_IID_DIFF, DATA_DISTRIBUTION_STANDARD_IID_HOMO, DATA_DISTRIBUTION_REAL_FEMNIST
from utils.model_variation_generator import bert_model_variation
from validators.config_validator import ConfigValidator

#from datasets import Dataset as HFDataset

def load_and_prepare_data(filepath):
    data = pd.read_csv(filepath)
    data.dropna(inplace=True)
    data['class_index'] = pd.to_numeric(data['class_index'], errors='coerce').astype(int)
    data['text'] = data['question_title'] + ' ' + data['question_content'] + ' ' + data['best_answer']
    data.drop(columns=["question_content", "question_title", "best_answer"], inplace=True)  # Keep only necessary columns
    return data

def tokenize_data(data, tokenizer, max_len=256):
    return tokenizer(
        list(data["text"]),
        truncation=True,
        padding="max_length",
        max_length=max_len,
        return_tensors=None
    )

class TextDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)



def split_data_random_nonoverlapping(df, n_clients, n_classes_per_client, seed=42):
    np.random.seed(seed)

    # All unique class labels
    print("---------------------------------------------------------------------")
    print(df.columns)  # Print first row properly using iloc

    class_labels = df['class_index'].unique()
    total_classes = len(class_labels)

    if n_classes_per_client > total_classes:
        raise ValueError("n_classes_per_client cannot be more than total number of classes.")

    # Step 1: Randomly assign each client n_classes_per_client classes
    client_class_map = {}
    for client_id in range(n_clients):
        client_class_map[client_id] = np.random.choice(class_labels, size=n_classes_per_client, replace=False)

    # Step 2: Create reverse map: which clients are assigned to each class
    class_client_map = defaultdict(list)
    for client_id, class_list in client_class_map.items():
        for cls in class_list:
            class_client_map[cls].append(client_id)

    # Step 3: Group samples by class
    class_to_samples = {cls: df[df['class_index'] == cls].sample(frac=1.0, random_state=seed).reset_index(drop=True)
                        for cls in class_labels}

    # Step 4: Distribute each class's samples to assigned clients
    client_data = defaultdict(list)

    for cls, samples in class_to_samples.items():
        assigned_clients = class_client_map[cls]
        n_samples = len(samples)

        if len(assigned_clients) == 0:
            # If no clients are assigned to this class, skip it
            continue

        # Random proportions for each client assigned to this class
        proportions = np.random.dirichlet(np.ones(len(assigned_clients)))
        counts = (proportions * n_samples).astype(int)

        # Adjust to make sure sum(counts) == n_samples
        while counts.sum() < n_samples and len(counts) > 0:
            counts[np.random.randint(0, len(counts))] += 1
        while counts.sum() > n_samples and len(counts) > 0:
            idx = np.where(counts > 0)[0]
            if len(idx) > 0:
                counts[np.random.choice(idx)] -= 1

        # Distribute the data
        start_idx = 0
        for client_id, count in zip(assigned_clients, counts):
            if count > 0:  # Only add data if there are samples to add
                client_data[client_id].append(samples.iloc[start_idx:start_idx + count])
            start_idx += count

    # Step 5: Concatenate client dataframes
    for client_id in client_data:
        if len(client_data[client_id]) > 0:  # Only concatenate if there's data
            client_data[client_id] = pd.concat(client_data[client_id]).reset_index(drop=True)
        else:
            # If no data was assigned to this client, create an empty DataFrame with the same columns
            client_data[client_id] = pd.DataFrame(columns=df.columns)

    return client_data, client_class_map


def yahoo_factory(
        data_path: str,
        config: 'ConfigValidator',
        max_length: int = 512,
        pad_to_max: bool = True,
        shuffle: bool = True,
        test_fraction: float=0.2
):

    data_path = data_path + '/yahoo/yahoo.csv'
    df = load_and_prepare_data(data_path)

    n_clients = config.NUMBER_OF_CLIENTS

    if config.DATASET_TYPE not in [DATA_DISTRIBUTION_REAL_FEMNIST, DATA_DISTRIBUTION_STANDARD_IID_HOMO, DATA_DISTRIBUTION_STANDARD_IID_DIFF, DATA_DISTRIBUTION_STANDARD_NON_IID_DIR] :
        n_classes_per_client = int(config.DATA_DISTRIBUTION[-1])
    else :
        raise ValueError (f"{config.DATA_DISTRIBUTION} has not developed yet!")

    train_df, test_df, train_labels, test_labels = train_test_split(
        df['text'], df['class_index'], test_size=test_fraction, random_state=42, stratify=df['class_index']
    )

    # Create DataFrames with both text and class_index
    train_df = pd.DataFrame({'text': train_df, 'class_index': train_labels})
    test_df = pd.DataFrame({'text': test_df, 'class_index': test_labels})

    client_train_data, client_class_map = split_data_random_nonoverlapping(
        train_df, n_clients=n_clients, n_classes_per_client=n_classes_per_client
    )

    tokenizer = BertTokenizer.from_pretrained(bert_model_variation(config.TRANSFORMER_MODEL_SIZE))
    train_loaders = []
    test_loaders = []

    for client_id in range(n_clients):
        print(f"processing client: {client_id} train")
        client_df = client_train_data[client_id]
        encodings = tokenize_data(client_df, tokenizer, max_len=max_length)
        labels = list(client_df['class_index'])
        train_dataset = TextDataset(encodings, labels)
        train_loader = DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=shuffle)
        train_loaders.append(train_loader)

    test_df_shuffled = test_df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    val_split = np.array_split(test_df_shuffled, n_clients)

    for client_id in range(n_clients):
        print(f"processing client: {client_id} test")

        client_test_df = val_split[client_id].reset_index(drop=True)
        test_encodings = tokenize_data(client_test_df, tokenizer, max_len=max_length)
        test_labels = list(client_test_df['class_index'])
        test_dataset = TextDataset(test_encodings, test_labels)
        test_loader = DataLoader(test_dataset, batch_size=config.TEST_BATCH_SIZE, shuffle=shuffle)
        test_loaders.append(test_loader)

    return train_loaders, test_loaders
