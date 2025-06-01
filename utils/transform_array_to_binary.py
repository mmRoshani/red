def transform_array(input_array):
    """
    Transforms an array by replacing non-zero numbers with 1 and keeping 0s as 0.

    Args:
      input_array: A list containing numbers.

    Returns:
      A new list of the same size, where non-zero elements are replaced by 1,
      and zero elements remain 0.
    """
    return [1 if item != 0 else 0 for item in input_array]
