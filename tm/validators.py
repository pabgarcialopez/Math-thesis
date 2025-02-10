# tm/validators.py

from functools import wraps

def validate_params(validator_func):
    """A decorator to validate method parameters before execution."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            validator_func(self, *args, **kwargs)
            return func(self, *args, **kwargs)
        return wrapper
    return decorator

def validate_transition_params(self, state, symbol, next_state, write_symbol, direction):
    """
    Validate parameters for a transition.
    - 'direction' must be either 'L' or 'R'.
    - 'state' and 'next_state' must be within the allowed range.
    - 'symbol' must be either one of the input symbols or the blank symbol.
    """
    if direction not in {'L', 'R'}:
        raise ValueError("Direction must be 'L' (left) or 'R' (right).")
    if state >= self.num_states or next_state >= self.num_states:
        raise ValueError("State or next_state exceeds number of states.")
    if symbol not in self.input_symbols and symbol != self.blank_symbol:
        raise ValueError("Invalid input symbol.")

def validate_binary_input(binary_input, bits=5):
    """
    Validate that binary_input is a string of exactly `bits` characters,
    each either '0' or '1'. If binary_input is None, no error is raised.
    """
    if binary_input is None:
        return
    if not isinstance(binary_input, str):
        raise ValueError("Binary input must be a string.")
    if len(binary_input) != bits or not set(binary_input).issubset({'0', '1'}):
        raise ValueError(f"Binary input must be exactly {bits} bits (e.g., '10101').")
