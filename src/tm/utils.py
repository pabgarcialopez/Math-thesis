import random
from src.tm.machine import TuringMachine

# --------------------------------------------------------------------------------------------------
# GENERATORS for the Turing Machine
# --------------------------------------------------------------------------------------------------

def generate_random_input(tape_length):
    return ''.join(random.choice('01') for _ in range(tape_length))

def generate_random_transitions(turing_machine):
    trans_prob = turing_machine.trans_prob
    num_states = 2 ** turing_machine.state_bits

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

def generate_turing_machines(num_machines, tape_bits, head_bits, state_bits, trans_prob):
    return [
        TuringMachine(
            tape_bits=tape_bits,
            head_bits=head_bits,
            state_bits=state_bits,
            trans_prob=trans_prob)

        for _ in range(num_machines)
    ]

# --------------------------------------------------------------------------------------------------
# GETTERS for the Turing Machine
# --------------------------------------------------------------------------------------------------

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
    
def get_projected_history_function(config_history, projection):
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
        "tape_bits": tm.tape_bits,
        "head_bits": tm.head_bits,
        "state_bits": tm.state_bits,
        "trans_prob": tm.trans_prob,
        "outcome": tm.outcome,
        "config_history": list(tm.config_history)
    }