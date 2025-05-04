def get_last_char_as_int(input_string):
  """
  Gets the last character of a string and converts it to an integer.

  Args:
    input_string: The input string.

  Returns:
    The last character of the string as an integer.
    Returns None if the string is empty or the last character is not a digit.
  """
  if not input_string:
    return None
  last_char = input_string[-1]
  if last_char.isdigit():
    return int(last_char)
  else:
    return None
