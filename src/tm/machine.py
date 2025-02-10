# tm/machine.py

from src.tm.validators import validate_params, validate_transition_params, validate_binary_input

class TuringMachine:
    def __init__(self, num_states=4, input_symbols={'0', '1'}, blank_symbol='_',
                 transition_function=None, initial_head_position=1, accepting_states=None, 
                 binary_input=None, trans_prob=0.5, debug=False):
        """
        Initialize the Turing Machine.
        The tape is composed of 7 cells:
          - Index 0: left wall (fixed)
          - Indexes 1-5: variable tape content (5 bits)
          - Index 6: right wall (fixed)
        The head (cursor) can be in any of the 7 positions (0 to 6), using 3 bits.
        The current state is represented with 2 bits (supporting up to 4 states).

        If binary_input is provided, it must be exactly 5 bits (e.g., '10101')
        and will be loaded into the variable region of the tape (cells 1 to 5).
        Otherwise, the variable region is initialized with the blank symbol.

        If accepting_states is not provided, they are generated randomly.
        If transition_function is not provided, it is generated randomly with probability `trans_prob`.
        """
        if blank_symbol in input_symbols:
            raise ValueError("Blank symbol cannot be an input symbol.")
        
        self.num_states = num_states
        self.input_symbols = input_symbols
        self.blank_symbol = blank_symbol

        # Validate binary_input
        validate_binary_input(binary_input, bits=5)
        
        if binary_input is not None:
            self.binary_input = binary_input
            variable_cells = list(binary_input)
        else:
            self.binary_input = None
            variable_cells = [blank_symbol] * 5

        # Initialize tape with boundaries (walls) and 5 variable cells.
        self.tape = ['|'] + variable_cells + ['|']
        
        if initial_head_position < 0 or initial_head_position > 6:
            raise ValueError("Initial head position must be between 0 and 6.")
        self.initial_head_position = initial_head_position
        self.head_position = initial_head_position
        
        self.current_state = 0  # Start in state 0
        
        # Generate accepting states if not provided.
        if accepting_states is None:
            from src.tm.generators import generate_random_accepting_states
            accepting_states = generate_random_accepting_states(num_states=self.num_states)
        self.accepting_states = accepting_states
        
        # Generate transition function if not provided.
        if transition_function is None:
            from src.tm.generators import generate_random_transitions
            transition_function = generate_random_transitions(self, trans_prob=trans_prob)
        self.transition_function = transition_function
        
        # List to record the history of configurations (each as a 10-bit string).
        self.config_history = []
        
        # Debug flag.
        self.debug = debug

    def log(self, message):
        """Helper method to print debug messages if debugging is enabled."""
        if self.debug:
            print(message)
    
    @validate_params(validate_transition_params)
    def add_transition(self, state, symbol, next_state, write_symbol, direction):
        """Add a transition to the TM's transition function."""
        self.transition_function[(state, symbol)] = (next_state, write_symbol, direction)
    
    def print_transitions(self):
        """Print the complete transition function in a readable format."""
        print("\nTransition Function:")
        if not self.transition_function:
            print("  No transitions defined.")
        else:
            for (state, symbol), (next_state, write_symbol, direction) in sorted(self.transition_function.items()):
                print(f"  (state {state}, symbol '{symbol}') -> (state {next_state}, write '{write_symbol}', move {direction})")
    
    def read_current_symbol(self):
        """Read the symbol currently under the head."""
        return self.tape[self.head_position]
    
    def get_transition(self, current_symbol):
        """Return the transition for (current_state, current_symbol), if any."""
        return self.transition_function.get((self.current_state, current_symbol))
    
    def write_symbol(self, write_symbol):
        """Write a symbol to the current head position on the tape."""
        self.tape[self.head_position] = write_symbol
    
    def move_head(self, direction):
        """
        Move the head left or right, ensuring it does not leave the 7-cell tape.
        Note: The head is constrained to the variable region (cells 1 to 5).
        """
        if direction == 'R' and self.head_position < 5:
            self.head_position += 1
        elif direction == 'L' and self.head_position > 1:
            self.head_position -= 1
    
    def is_accepting(self):
        """Return True if the current state is an accepting state."""
        return self.current_state in self.accepting_states
    
    def get_configuration(self):
        """
        Return the current configuration as a 10-bit string composed of:
          - 5 bits for the tape content (cells 1 to 5)
          - 3 bits for the head position (0 to 6, in 3-bit binary)
          - 2 bits for the current state (assuming up to 4 states)
        """
        tape_bits = ''.join(self.tape[1:6])
        head_bits = format(self.head_position, '03b')
        state_bits = format(self.current_state, '02b')
        return tape_bits + head_bits + state_bits
    
    def get_projected_configuration(self, config_choice="final"):
        """
        Retorna la proyección de la configuración (los 5 bits de la cinta)
        a partir de la configuración elegida.
        
        Parámetros:
          config_choice: 'initial', 'middle' o 'final'
            - 'initial': toma la primera configuración registrada.
            - 'middle': toma la configuración del medio de la historia.
            - 'final' (por defecto): toma la última configuración.
        
        Si no hay historia registrada, retorna la proyección de la configuración
        actual.
        """
        # Asegurarse de tener un historial; si no, usar la configuración actual.
        if not self.config_history:
            config = self.get_configuration()
        else:
            if config_choice == "initial":
                config = self.config_history[0]
            elif config_choice == "middle":
                config = self.config_history[len(self.config_history) // 2]
            else:  # "final" por defecto
                config = self.config_history[-1]
        # Retornamos los 5 primeros bits (la parte de la cinta)
        return config[:5]
    
    def step(self):
        """
        Execute a single step (transition) of the Turing Machine.
        If no transition is defined for the current (state, symbol) pair,
        the machine halts and returns "accepted" or "rejected".
        """
        current_symbol = self.read_current_symbol()
        transition = self.get_transition(current_symbol)

        if not transition:
            return "accepted" if self.is_accepting() else "rejected"
        
        next_state, write_symbol, direction = transition
        self.write_symbol(write_symbol)
        self.move_head(direction)
        self.current_state = next_state
    
    def run(self, max_steps=1000):
        """
        Run the Turing Machine with a limit on the number of steps.
        Records each configuration (as a 10-bit string) in self.config_history.
        Returns "accepted", "rejected", or "inconclusive" if max_steps is reached.
        """
        self.config_history = []
        self.config_history.append(self.get_configuration())
        step_count = 0
        while step_count < max_steps:
            result = self.step()
            self.config_history.append(self.get_configuration())
            if result is not None:
                return result
            step_count += 1
        return "accepted" if self.is_accepting() else "rejected"
    
    def get_history_function(self):
        """
        Returns a vector (list) of 1024 entries (since configurations are 10 bits)
        representing the history function of the Turing Machine.
        The vector is indexed by the lexicographic order of 10-bit strings.
        For each configuration (as a 10-bit string) in this fixed order,
        the vector has a 1 if that configuration was encountered during execution,
        or 0 otherwise.
        """
        N = 10
        domain_size = 2 ** N  # 1024
        history_set = set(self.config_history)
        return [1 if format(i, '010b') in history_set else 0 for i in range(domain_size)]
    
    def print_tape(self, label="Tape:"):
        """
        Print the full tape (with boundaries) and indicate the head position.
        The tape is printed as: <label> <tape_contents>
        The caret '^' is printed on the next line, aligned so that it appears
        directly under the tape cell where the head is located.
        """
        tape_visual = ''.join(self.tape)
        line = f"{label} {tape_visual}"
        print(line)
        offset = len(label) + 1 + self.head_position
        print(" " * offset + "^")
    
    def print_history(self):
        """Print the recorded history of configurations (each as a 10-bit string)."""
        print("Configuration History (5 bits tape + 3 bits head + 2 bits state):")
        for config in self.config_history:
            print(config)
