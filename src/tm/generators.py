# tm/generators.py

import random
from src.tm.machine import TuringMachine

def generate_random_transitions(turing_machine, trans_prob=0.5):
    """
    Genera una función de transición aleatoria para la TuringMachine dada.
    Para cada estado y para cada símbolo posible (símbolos de entrada + símbolo en blanco),
    añadimos una transición con probabilidad `trans_prob`.
    """
    num_states = turing_machine.num_states
    input_symbols = turing_machine.input_symbols
    blank_symbol = turing_machine.blank_symbol
    directions = ['L', 'R']
    
    transition_function = {}
    for state in range(num_states):
        possible_symbols = list(input_symbols) + [blank_symbol]
        for symbol in possible_symbols:
            if random.random() < trans_prob:
                next_state = random.randint(0, num_states - 1)
                write_symbol = random.choice(list(input_symbols))
                direction = random.choice(directions)
                transition_function[(state, symbol)] = (next_state, write_symbol, direction)
    return transition_function

def generate_random_accepting_states(num_states=4):
    """
    Genera aleatoriamente una lista de estados aceptantes en el rango [0..num_states-1].
    Cada estado se elige con 50% de probabilidad de ser aceptante.
    """
    return [state for state in range(num_states) if random.choice([True, False])]

def generate_random_input(tape_length=5):
    """
    Genera una cadena binaria aleatoria de longitud `tape_length`.
    """
    return ''.join(random.choice('01') for _ in range(tape_length))

def generate_tm_input_pairs(n, trans_prob=0.5, tape_length=5, num_states=4):
    """
    Genera n TuringMachine objects, cada uno con una entrada binaria aleatoria de `tape_length` bits.
    La máquina generará sus estados aceptantes y su función de transición automáticamente.
    """
    machines = []
    for _ in range(n):
        binary_input = generate_random_input(tape_length)
        tm = TuringMachine(
            tape_length=tape_length,
            num_states=num_states,
            binary_input=binary_input,
            trans_prob=trans_prob
        )
        machines.append(tm)
    return machines
