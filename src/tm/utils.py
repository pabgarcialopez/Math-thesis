import random
from typing import List
from src.tm.machine import TuringMachine
from src.experiments.config import DEFAULT_TRANSITION_PROBABILITY


# --------------------------------------------------------------------------------------------------
# GENERATORS for the Turing Machine
# --------------------------------------------------------------------------------------------------

def generate_turing_machine(*, config, binary_input, transition_function):
    return TuringMachine(
            config=config,
            binary_input=binary_input,
            transition_function=transition_function)
    
def generate_turing_machines(*, config, binary_inputs=None, transition_probability=None, transition_functions=None, num_machines):
    
    turing_machines = []
    if binary_inputs is None:
        binary_inputs = [generate_random_binary_input(config['tape_bits']) for _ in range(num_machines)] 
        
    if transition_probability is None:
        transition_probability = DEFAULT_TRANSITION_PROBABILITY
        
    if transition_functions is None:
        transition_functions = [
            generate_random_transitions(
                transition_probability=transition_probability,
                state_bits=config['state_bits']) 
            for _ in range(num_machines)]
        
    for i in range(num_machines):        
        turing_machines.append(
            generate_turing_machine(
                config=config,
                binary_input=binary_inputs[i],
                transition_function=transition_functions[i]) 
        )
        
    return turing_machines
    
def generate_random_binary_input(tape_length):
    return ''.join(random.choice('01') for _ in range(tape_length))

def generate_random_transitions(*, transition_probability, state_bits):
    trans_prob = transition_probability
    num_states = 2 ** state_bits

    symbols = ['0', '1']
    directions = ['L', 'R']
    
    transition_function = {}
    for state in range(num_states):
        for symbol in symbols:
            if random.random() < trans_prob:
                next_state = random.randint(0, num_states - 1)
                write_symbol = random.choice(symbols)
                direction = random.choice(directions)
                transition_function[(state, symbol)] = (next_state, write_symbol, direction)
    return transition_function

# --------------------------------------------------------------------------------------------------
# GETTERS for the Turing Machine
# --------------------------------------------------------------------------------------------------

def get_num_steps(turing_machine):
    return turing_machine.num_steps

def get_configuration(turing_machine):
    """
    Returns a binary string representing the current configuration of a Turing Machine,
    zero-padded so that the head and state each occupy a fixed number of bits.
    """

    # Get necessary info
    tape = turing_machine.tape
    head = turing_machine.head_position
    state = turing_machine.current_state

    # Paddings so that, for instance, 2 = "010" if head_bits = 3 rather than "10"
    head_padding = turing_machine.head_bits
    state_padding = turing_machine.state_bits

    tape_bits = ''.join(tape)
    head_bits = format(head, f'0{head_padding}b')
    state_bits = format(state, f'0{state_padding}b')
    return tape_bits + head_bits + state_bits
    
def get_projected_history_function(config_history: List[str], projection: List[int]):
    """
    Returns the projected history function of a Turing Machine's conf_history, by building
    a list of length `2^(len(projection))`, where each position is indexed by 
    the integer value of the projected configuration bits. Each position is set
    to 1 if that projected pattern was seen during the execution of the Turing Machine,
    or 0 otherwise.
    """

    domain_size = 2 ** len(projection)
    projected_history_func = [0] * domain_size

    for config in config_history:
        projected_config = "".join(config[index] for index in projection)
        idx = int(projected_config, 2)
        projected_history_func[idx] = 1

    # Need to reverse it because metrics expect
    # that the entry 0 corresponds to 1...1 and
    # the last entry corresponds to 0...0.
    projected_history_func.reverse()
    return projected_history_func


def get_history_function(turing_machine):
    """
    Returns the history function of a Turing Machine, by building a list of
    of length `2^(config_bits)`, marking 1 for each visited configuration
    and 0 otherwise.
    """
    projection = list(range(turing_machine.config_bits))
    return get_projected_history_function(turing_machine.config_history, projection)

# --------------------------------------------------------------------------------------------------
# SERIALIZERS for the Turing Machine
# --------------------------------------------------------------------------------------------------

def serialize_turing_machine(tm):
    return {
        "config": tm.config,
        "binary_input": tm.binary_input,
        "num_steps": tm.num_steps,
        "outcome": tm.outcome,
        "history_function": get_history_function(tm),
        "config_history": list(tm.config_history)
    }    
