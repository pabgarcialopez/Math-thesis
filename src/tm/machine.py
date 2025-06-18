

class TuringMachine:
    def __init__(self, *, config, binary_input, transition_function):

        self.config = config
        self.tape_bits = config['tape_bits']
        self.head_bits = config['head_bits']
        self.state_bits = config['state_bits']
        
        # Initialize tape
        self.binary_input = binary_input
        self.tape = list(binary_input)

        # Initialize machine's internal info
        self.outcome = None
        self.num_steps = 0
        self.head_position = 0
        self.current_state = 0
        
        # Use passed transition function if any. Otherwise, generate a random one
        self.transition_function = transition_function

        # Initialize the configuration history
        from src.tm.utils import get_configuration
        self.config_history = set([get_configuration(self)])
        self.config_bits = self.tape_bits + self.head_bits + self.state_bits


    def move_head(self, direction):
        # Head and tape might not align, so we need to bound the head
        limit = min(len(self.tape) - 1, 2 ** self.head_bits - 1)
        if direction == 'R' and self.head_position < limit:
            self.head_position += 1
        elif direction == 'L' and self.head_position > 0:
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
        Runs the Turing Machine, recording configurations in self.config_history,
        until it halts or enters a loop.
        """
        from src.tm.utils import get_configuration
        
        while True:
            result = self.step()
            current_config = get_configuration(self)
            if result is not None: # Machine halted
                self.config_history.add(current_config)
                self.outcome = "halt"  
                break
            if current_config in self.config_history: # Entering a loop
                self.outcome = "loop"
                break
            self.config_history.add(current_config)  
