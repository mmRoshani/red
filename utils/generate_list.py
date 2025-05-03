from utils.log import Log
from typing import List

def generate_list(count: int, log: Log) -> List[int]:
    """Generates a list of string representations of integers starting from zero.

    Args:
      count: The number of IDs to generate.
      log: application log instance

    Returns:
      A list of strings representing integers from 0 up to (but not including) count.
    """

    generated_list =  list(range(count))

    log.info(f'generated a list for ids from 0 to {count} with length of {generated_list}')

    return generated_list