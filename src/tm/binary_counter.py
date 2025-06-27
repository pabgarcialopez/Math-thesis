from pyeda.inter import exprvars, truthtable # type: ignore
from pyeda.boolalg.minimization import espresso_tts # type: ignore
from src.experiments.utils.computing import generate_binary_counter_transitions, generate_alternating_counter_transitions, measure_minimal_dnf

class BinaryCounter:
    def __init__(self, config, alternating=False):
        
        self.config = config
        self.tape_bits = config['tape_bits']
        self.head_bits = config['head_bits']
        self.state_bits = config['state_bits']
                
        self.LEFT_WALL = "L*"
        self.RIGHT_WALL = "R*"
        
        self.binary_input = '0' * self.tape_bits
        
        # Tape with left and right symbols to bound it
        self.tape = [self.LEFT_WALL] + list(self.binary_input) + [self.RIGHT_WALL]

        # Initialize machine's internal info
        self.outcome = None
        self.num_steps = 0
        self.head_position = 0
        self.current_state = 0
        
        # Build the transitions
        if alternating: self.transition_function = generate_alternating_counter_transitions()
        else: self.transition_function = generate_binary_counter_transitions()

        # Initialize config history with initial configuration
        self.config_history = [self._get_configuration()]
        self.config_bits = self.tape_bits + self.head_bits + self.state_bits
        
    def _get_configuration(self):
        """
        Returns a binary string representing the current configuration of a Turing Machine,
        zero-padded so that the head and state each occupy a fixed number of bits.
        """
        
        # Get necessary info
        tape = self.tape
        head = self.head_position
        state = self.current_state
        
        # Adjust head if necessary so that it doesn't point to the walls and is in the range [0, ..., tape_bits - 1]
        #                          0  1 2 3 4  5  
        # For instance, if tape is L* _ _ _ _ R*
        if head == 0:
            head = 0
        elif head == len(self.tape) - 1:
            head -= 2
        else: # [1, 2, 3, 4]
            head -= 1

        # Paddings so that, for instance, 2 = "010" if head_bits = 3 rather than "10"
        head_padding = self.head_bits
        state_padding = self.state_bits

        tape_bits = ''.join(tape[1:-1]) # Leave out wall symbols
        head_bits = format(head, f'0{head_padding}b')
        state_bits = format(state, f'0{state_padding}b')
        return tape_bits + head_bits + state_bits
        

    def move_head(self, direction):        
        if direction == 'R':
            self.head_position += 1
        elif direction == 'L':
            self.head_position -= 1
        
    def step(self):
        """
        Executes a single transition step. If no transition is found for the current (state, symbol),
        the machine halts.
        """
        current_symbol = self.tape[self.head_position]
        transition = self.transition_function.get((self.current_state, current_symbol))
        
        if not transition:
            return "halt"
        
        next_state, write_symbol, direction = transition
        self.current_state = next_state
        self.tape[self.head_position] = write_symbol
        self.move_head(direction)
        self.num_steps += 1
            
    def run(self):
        """
        Runs the Turing Machine until it naturally halts (no transition),
        registrando todas las configuraciones vistas.
        """
        while True:
            result = self.step()
            if result is not None:
                self.outcome = "halt"
                self.config_history.append(self._get_configuration())
                break
            self.config_history.append(self._get_configuration())