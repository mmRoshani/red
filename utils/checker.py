def device_checker(device: str | None) -> str:
    if device is None:
        return 'cpu'
    return device


def none_checker(value, value_name):
    if value is None:
        raise ValueError(f"none value is not acceptable for value {value_name}")
    return value
