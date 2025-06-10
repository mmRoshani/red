import datetime
from dataclasses import dataclass
from dataclasses import field
from typing import Dict

@dataclass
class Message:
    header: str = None
    sender_id: str = None
    timestamp: datetime.datetime = field(default_factory=datetime.datetime.now)
    body: Dict = field(default_factory=dict)
