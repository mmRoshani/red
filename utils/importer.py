from importlib import import_module
from typing import Any

def dynamic_import(dot_path: str, class_name: str) -> Any:
    name = dot_path.strip('.py')
    imp = import_module(name)
    model = getattr(imp, class_name)
    return model()
