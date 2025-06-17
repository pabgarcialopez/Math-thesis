from pprint import pprint
from pyeda.inter import exprvars, truthtable # type: ignore
from pyeda.boolalg.minimization import espresso_tts # type: ignore
import numpy as np
from src.tm.utils import get_history_function

class BinaryCounter:
    def __init__(self, tape_bits, head_bits, state_bits):


        self.tape_bits = tape_bits
        self.head_bits = head_bits
        self.state_bits = state_bits
        
        self.LEFT_WALL = "L*"
        self.RIGHT_WALL = "R*"
        
        # Tape with left and right symbols to bound it
        self.tape = [self.LEFT_WALL] + ['0' for _ in range(tape_bits)] + [self.RIGHT_WALL]

        # Initialize machine's internal info
        self.outcome = None
        self.num_steps = 0
        self.head_position = 0
        self.current_state = 0
        
        # Build the transitions
        self.transition_function = self._build_transitions()

        # Initialize config history with initial configuration
        self.config_history = set([self._get_configuration()])
        self.config_bits = tape_bits + head_bits + state_bits
        
    def _get_configuration(self):
        """
        Returns a binary string representing the current configuration of a Turing Machine,
        zero-padded so that the head and state each occupy a fixed number of bits.
        """
        
        # assert 1 <= self.head_position
        # assert self.head_position <= self.tape_bits 
        

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
        
    def _build_transitions(self):
        # transition_function[(state, symbol)] = (next_state, write_symbol, direction)
        transition_function = {}
        
        # State 0 transitions
        transition_function[(0, 'L*')] = (1, 'L*', 'R')
        
        # State 1 transitions
        transition_function[(1, '0')] = (0, '1', 'L')
        transition_function[(1, '1')] = (2, '0', 'R')
        
        # State 2 transitions
        transition_function[(2, '1')] = (2, '0', 'R')
        transition_function[(2, '0')] = (3, '1', 'L')
        
        # State 3 transitions
        transition_function[(3, '0')] = (3, '0', 'L')
        transition_function[(3, 'L*')] = (1, 'L*', 'R')
        
        return transition_function

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
        Runs the Turing Machine, recording configurations in self.config_history,
        until it halts or enters a loop.
        """
                
        while True:
            result = self.step()
            current_config = self._get_configuration()
            if result is not None: # Machine halted
                # self.config_history.add(current_config)
                self.outcome = "halt"  
                break
            if current_config in self.config_history: # Entering a loop
                self.outcome = "loop"
                break
            self.config_history.add(current_config)  
            
class DebugBC(BinaryCounter):
    def run(self):
        step = 0
        # print(self._get_configuration())
        while True:
            result = self.step()
            cfg = self._get_configuration()
            # print(f"Step {step:2d}: tape={''.join(self.tape[1:-1])}  head={self.head_position}  "
            #       f"state={self.current_state}  config={cfg}")
            step += 1
            if result is not None:
                # print("  ⇒ halted")
                break
            # if cfg in self.config_history:
            #     print("  ⇒ loop detected on cfg repeat")
            #     break
            self.config_history.add(cfg)
            
def measure_minimal_dnf(bool_vector):
    """
    Given a boolean vector (list of 0/1 or False/True) of length 2^n,
    build a PyEDA truthtable, run espresso_tts,
    and return (num_terms, total_literals) for the minimal DNF expression.
    """
    n = int(np.log2(len(bool_vector)))
    xs = exprvars('x', n)
    bool_tuple = tuple(bool(x) for x in bool_vector)

    tt = truthtable(xs, bool_tuple)
    min_exprs = espresso_tts(tt)
    if not min_exprs:
        return 0, 0
    expr = min_exprs[0]

    ast = expr.to_ast()
    if isinstance(ast, tuple) and ast[0] == 'or':
        terms = ast[1:]
    else:
        terms = [ast]

    num_terms = len(terms)
    total_literals = 0
    for term in terms:
        if isinstance(term, tuple) and term[0] == 'and':
            total_literals += len(term) - 1
        else:
            total_literals += 1
    return num_terms, total_literals

if __name__ == "__main__":
    
    configs = [
        # {"tape_bits": 2, "head_bits": 1, "state_bits": 2},
        # {"tape_bits": 3, "head_bits": 1, "state_bits": 2},
        # {"tape_bits": 3, "head_bits": 2, "state_bits": 2},
        # {"tape_bits": 4, "head_bits": 2, "state_bits": 2},
        # {"tape_bits": 5, "head_bits": 2, "state_bits": 2},
        # {"tape_bits": 5, "head_bits": 3, "state_bits": 2},
        # {"tape_bits": 6, "head_bits": 2, "state_bits": 2},
        # {"tape_bits": 6, "head_bits": 3, "state_bits": 2},
        # {"tape_bits": 7, "head_bits": 3, "state_bits": 2},
        # {"tape_bits": 8, "head_bits": 3, "state_bits": 2},
        # {"tape_bits": 9, "head_bits": 3, "state_bits": 2},
        # {"tape_bits": 10, "head_bits": 3, "state_bits": 2},
        # {"tape_bits": 11, "head_bits": 3, "state_bits": 2},
        # {"tape_bits": 12, "head_bits": 3, "state_bits": 2},
        {"tape_bits": 11, "head_bits": 4, "state_bits": 2},
        {"tape_bits": 12, "head_bits": 4, "state_bits": 2},
        # {"tape_bits": 16, "head_bits": 4, "state_bits": 2},
    ]
    
    results = []
    for i, config in enumerate(configs):
        print(f"Executing config {i + 1}...")
        tape_bits = config["tape_bits"]
        head_bits = config["head_bits"]
        state_bits = config["state_bits"]
        
        total_bits = tape_bits + head_bits + state_bits
        db = DebugBC(tape_bits=tape_bits, head_bits=head_bits, state_bits=state_bits)
        db.run()

        # Get history function
        history_func = get_history_function(db)
        num_minterms, num_literals = measure_minimal_dnf(history_func)

        result = {
            "total_bits": total_bits,
            "num_minterms": num_minterms,
            "num_literals": num_literals,
        }
        
        results.append(result)
        
    for result in results:
        print(f"Total bits: {result['total_bits']}")
        print(f"Num minterms: {result['num_minterms']}, num literals: {result['num_literals']}")
    
    


