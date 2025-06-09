from dataclasses import dataclass
from typing import List
from utils.log import Log
from torch.utils.data import DataLoader


@dataclass
class RuntimeConfig:

    clients_id_list: List[str] | List[int]

    train_loaders: List[DataLoader]
    test_loaders: List[DataLoader]

    # TODO: HE related runtime configurations
    
    log: Log
