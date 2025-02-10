# tm/generators.py

import random
from tm.machine import TuringMachine

def generate_random_transitions(turing_machine, trans_prob=0.5):
    """
    Generate a random transition function for the given Turing Machine.
    For each state and for each possible symbol (input symbols + blank symbol),
    we add a transition with probability `trans_prob`.
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
    Generate a random list of accepting states from 0 to num_states-1.
    Each state is chosen with a 50% probability of being accepting.
    """
    return [state for state in range(num_states) if random.choice([True, False])]

def generate_random_input(bits=5):
    """
    Generate a random binary string of the given number of bits.
    """
    return ''.join(random.choice('01') for _ in range(bits))

def generate_tm_input_pairs(n, trans_prob=0.5):
    """
    Generate n TuringMachine objects, each with a random 5-bit input.
    The machine will auto-generate its accepting states and transition function.
    """
    machines = []
    for _ in range(n):
        binary_input = generate_random_input(5)
        # The machine will generate accepting_states and transitions automatically.
        tm = TuringMachine(binary_input=binary_input, trans_prob=trans_prob)
        machines.append(tm)
    return machines
