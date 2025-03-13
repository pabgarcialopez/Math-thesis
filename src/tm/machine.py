# tm/machine.py

import math
from src.tm.validators import validate_params, validate_transition_params, validate_binary_input

class TuringMachine:
    def __init__(
        self,
        tape_length: int,
        num_states: int,
        total_bits: int = None,
        input_symbols={'0', '1'},
        blank_symbol='_',
        initial_head_position=1,
        transition_function=None,
        accepting_states=None,
        binary_input=None,
        trans_prob=0.5,
        debug=False
    ):
        """
        MUST pass total_bits.
        Then we do:
          state_bits = ceil(log2(num_states))
          head_position_bits = total_bits - tape_length - state_bits
          total_config_bits = total_bits

        If total_bits is None, we raise an error.
        """
        if total_bits is None:
            raise ValueError("TuringMachine requires total_bits.")
        if blank_symbol in input_symbols:
            raise ValueError("Blank symbol cannot be an input symbol.")
        
        self.num_states = num_states
        self.tape_length = tape_length
        self.total_bits_forced = total_bits
        self.input_symbols = input_symbols
        self.blank_symbol = blank_symbol
        self.binary_input = binary_input
        
        # Validate binary_input
        validate_binary_input(binary_input, tape_length)
        
        # Build tape: left wall + variable cells + right wall
        if binary_input is not None:
            variable_cells = list(binary_input)
        else:
            variable_cells = [blank_symbol] * tape_length
        self.tape = ['|'] + variable_cells + ['|']

        self.head_position = initial_head_position
        self.current_state = 0

        # If accepting_states not passed, generate randomly
        if accepting_states is None:
            from src.tm.generators import generate_random_accepting_states
            accepting_states = generate_random_accepting_states(num_states=self.num_states)
        self.accepting_states = accepting_states

        # If transition_function not passed, generate randomly
        if transition_function is None:
            from src.tm.generators import generate_random_transitions
            transition_function = generate_random_transitions(self, trans_prob=trans_prob)
        self.transition_function = transition_function

        self.state_bits = max(1, math.ceil(math.log2(num_states))) if num_states > 1 else 1
        self.head_position_bits = total_bits - tape_length - self.state_bits
        if self.head_position_bits < 1:
            raise ValueError(
                f"Invalid total_bits={total_bits}: tape_length={tape_length} "
                f"and state_bits={self.state_bits} => head_position_bits < 1."
            )

        self.total_config_bits = total_bits
        self.config_history = []
        self.debug = debug

    @validate_params(validate_transition_params)
    def add_transition(self, state, symbol, next_state, write_symbol, direction):
        self.transition_function[(state, symbol)] = (next_state, write_symbol, direction)

    def step(self):
        current_symbol = self.tape[self.head_position]
        transition = self.transition_function.get((self.current_state, current_symbol))
        if not transition:
            return "accepted" if self.current_state in self.accepting_states else "rejected"
        next_state, write_symbol, direction = transition
        self.tape[self.head_position] = write_symbol
        self.move_head(direction)
        self.current_state = next_state

    def run(self, max_steps=1000):
        self.config_history = [self.get_configuration()]
        for _ in range(max_steps):
            result = self.step()
            self.config_history.append(self.get_configuration())
            if result is not None:
                return result
        return "accepted" if self.current_state in self.accepting_states else "rejected"

    def move_head(self, direction):
        """
        Limit the head to [1..max_head_pos],
        where max_head_pos = (2^head_position_bits - 1).
        That ensures we never produce a head position that can't
        be represented in head_position_bits bits.
        """
        max_head_pos = (1 << self.head_position_bits) - 1

        if direction == 'R' and self.head_position < max_head_pos:
            self.head_position += 1
        elif direction == 'L' and self.head_position > 1:
            self.head_position -= 1

    
    def read_current_symbol(self):
        """Lee el símbolo bajo el cabezal."""
        return self.tape[self.head_position]
    
    def write_symbol(self, write_symbol):
        """Escribe un símbolo en la posición actual del cabezal."""
        self.tape[self.head_position] = write_symbol
    
    def is_accepting(self):
        """Devuelve True si el estado actual está en los estados aceptantes."""
        return self.current_state in self.accepting_states
    
    def get_transition(self, current_symbol):
        """Devuelve la transición para (current_state, current_symbol), si existe."""
        return self.transition_function.get((self.current_state, current_symbol))
    
    def get_configuration(self):
        tape_bits = ''.join(self.tape[1:1 + self.tape_length])
        head_bits = format(self.head_position, f'0{self.head_position_bits}b')
        state_bits = format(self.current_state, f'0{self.state_bits}b')
        return tape_bits + head_bits + state_bits
    
    def get_history_function(self):
        domain_size = 2 ** self.total_config_bits
        visited = set(self.config_history)
        vec = [0]*domain_size
        for cfg in visited:
            idx = int(cfg, 2)
            vec[idx] = 1
        return vec

    def get_projected_history_function(self):
        domain_size = 2 ** self.tape_length
        visited = set(cfg[:self.tape_length] for cfg in self.config_history)
        vec = [0]*domain_size
        for i in range(domain_size):
            pattern = format(i, f'0{self.tape_length}b')
            if pattern in visited:
                vec[i] = 1
        return vec
    
    def log(self, message):
        """Muestra un mensaje si el modo debug está activo."""
        if self.debug:
            print(message)
    
    def print_transitions(self):
        """
        Imprime la función de transición en un formato legible.
        """
        print("\nTransition Function:")
        if not self.transition_function:
            print("  No transitions defined.")
        else:
            for (state, symbol), (next_state, write_symbol, direction) in sorted(self.transition_function.items()):
                print(f"  (state {state}, symbol '{symbol}') -> (state {next_state}, write '{write_symbol}', move {direction})")
    
    def print_tape(self, label="Tape:"):
        """
        Imprime la cinta y una marca '^' bajo la posición del cabezal.
        """
        tape_visual = ''.join(self.tape)
        line = f"{label} {tape_visual}"
        print(line)
        offset = len(label) + 1 + self.head_position
        print(" " * offset + "^")
    
    def print_history(self):
        """Imprime el historial de configuraciones (cada una en binario)."""
        print("Configuration History:")
        for config in self.config_history:
            print(config)

