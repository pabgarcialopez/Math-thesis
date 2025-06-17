import random
from src.tm.machine import TuringMachine
from src.experiments.config import DEFAULT_TRANSITION_PROBABILITY


# --------------------------------------------------------------------------------------------------
# GENERATORS for the Turing Machine
# --------------------------------------------------------------------------------------------------

def _generate_turing_machine(*, config, binary_input, transition_function):
    return TuringMachine(
            config=config,
            binary_input=binary_input,
            transition_function=transition_function)
    
def generate_turing_machines(*, config, binary_inputs=None, transition_probability=None, transition_functions=None, num_machines):
    
    turing_machines = []
    if binary_inputs is None:
        binary_inputs = [generate_random_input(config['tape_bits']) for _ in range(num_machines)] 
        
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
            _generate_turing_machine(
                config=config,
                binary_input=binary_inputs[i],
                transition_function=transition_functions[i]) 
        )
        
    return turing_machines
    
def generate_random_input(tape_length):
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

def generate_long_tm_transitions(num_states, halting_fraction=0.1, seed=None):
    """
    Genera únicamente el diccionario de transiciones para una TM 
    que tarda "mucho" en parar, con estados que ocasionalmente llevan al estado de aceptación o parada.

    Parámetros:
    - num_states: total de estados (el último, num_states-1, es el estado de aceptación)
    - halting_fraction: fracción de estados no finales que pueden transicionar al estado de aceptación
    - seed: semilla para reproducibilidad (opcional)

    Devuelve:
    - transitions: dict con clave (state, symbol) y valor (write_symbol, move_dir, next_state). El estado de aceptación
    no define salidas. Cuando se llega a el, la ejecución de la MT termina.
    """
    if seed is not None:
        random.seed(seed)

    accept_state = num_states - 1
    non_final = list(range(num_states - 1))

    # Seleccionamos un subconjunto de estados con capacidad de halting
    k = max(1, int(len(non_final) * halting_fraction))
    halting_states = set(random.sample(non_final, k))

    transitions = {}

    # Creamos un orden aleatorio de no-final para construir un "ciclo"
    cycle = non_final[:]  
    random.shuffle(cycle)

    for i, s in enumerate(non_final):
        if s in halting_states:
            # Elegimos aleatoriamente el símbolo que disparará el halting
            halting_symbol = random.choice([0, 1])
            other_symbol = 1 - halting_symbol

            # Transición rara al estado de aceptación con movimiento aleatorio
            move_halting = random.choice(['L', 'R'])
            transitions[(s, halting_symbol)] = (halting_symbol, move_halting, accept_state)

            # Para el otro símbolo, seguimos en el ciclo de no-final
            write = random.choice([0, 1])
            move = random.choice(['L', 'R'])
            next_state = cycle[(i + 1) % len(cycle)]
            transitions[(s, other_symbol)] = (write, move, next_state)
        else:
            # Estados sin transición al estado de aceptación
            for sym in [0, 1]:
                write = random.choice([0, 1])
                move = random.choice(['L', 'R'])
                # Para uno usamos el siguiente en ciclo, para el otro uno aleatorio
                if sym == 0: next_state = cycle[(i + 1) % len(cycle)]  # noqa: E701
                else: next_state = random.choice(non_final)  # noqa: E701
                transitions[(s, sym)] = (write, move, next_state)

    return transitions

# Specific to experiment 6
def generate_turing_machines_with_transitions(num_machines, config, halting_fraction):
    num_states = 2 ** config['state_bits']
    transition_functions = [generate_long_tm_transitions(num_states, halting_fraction) for _ in range(num_machines)]
    turing_machines = []
    for transition_function in transition_functions:
        turing_machine = generate_turing_machine(config=config, transition_function=transition_function)
        turing_machines.append(turing_machine)
    return turing_machines



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
        # "tape_bits": tm.tape_bits,
        # "head_bits": tm.head_bits,
        # "state_bits": tm.state_bits,
        # "transition_probability": tm.trans_prob,
        "num_steps": tm.num_steps,
        "outcome": tm.outcome,
        "config_history": list(tm.config_history)
    }