import random
import numpy as np
from math import log2

from pyeda.inter import exprvars, truthtable # type: ignore
from pyeda.boolalg.minimization import espresso_tts # type: ignore

_VAR_CACHE = {}

def get_exprvars(n):
    if n not in _VAR_CACHE:
        _VAR_CACHE[n] = exprvars('x', n)
    return _VAR_CACHE[n]

def measure_minimal_dnf(bool_vector):
    # trivial cases
    s = sum(bool_vector)
    m = len(bool_vector)
    if s == 0:
        return 0, 0
    if s == m:
        n = int(log2(m))
        return 1, n

    # core work
    n = int(log2(m))
    xs = get_exprvars(n)
    bool_tuple = tuple(bool(x) for x in bool_vector)
    tt = truthtable(xs, bool_tuple)

    min_exprs = espresso_tts(tt)
    if not min_exprs:
        return 0, 0

    ast = min_exprs[0].to_ast()
    terms = ast[1:] if isinstance(ast, tuple) and ast[0]=='or' else [ast]
    num_terms = len(terms)
    total_literals = 0
    for t in terms:
        if isinstance(t, tuple) and t[0]=='and':
            total_literals += len(t)-1
        else:
            total_literals += 1
    return num_terms, total_literals

def equal_width_bins(data, n_bins):
    """
    Given a 1D sequence `data`, return an array of length n_bins+1
    containing equally‐spaced bin edges from min(data) to max(data).
    """
    arr = np.array(data, dtype=float)
    mn, mx = arr.min(), arr.max()
    # if all values identical, just return a trivial single‐bin
    if mn == mx:
        return np.array([mn, mn])
    return np.linspace(mn, mx, n_bins + 1)

import random

def generate_long_tm_transition_function(
    num_states: int,
    halting_fraction: float = 0.1,
) -> dict:
    """
    Build a transition function designed to make a Turing Machine run for many steps 
    before halting, by wiring most non-final states into a single cycle and only 
    allowing a small fraction of states to exit to the accept state.

    Parameters:
    - num_states (int): Total number of states; the last state (num_states-1) is the accept state.
    - halting_fraction (float): Fraction of non-final states that can transition to accept.

    Returns:
    - transitions (dict):
        Keys are (state, symbol) pairs. Values are tuples 
        (next_state, write_symbol, move_direction).
        The accept state has no outgoing transitions, so reaching it halts the machine.
    """
    accept_state = num_states - 1
    non_final = list(range(accept_state))

    # Pick which non-final states can actually halt
    k = max(1, int(len(non_final) * halting_fraction))
    halting_states = set(random.sample(non_final, k))

    # Build one big cycle through all non-final states
    cycle = non_final[:]
    random.shuffle(cycle)

    transitions = {}
    for i, s in enumerate(non_final):
        if s in halting_states:
            # Choose a unique halting symbol for this state
            halting_symbol = random.choice(['0', '1'])
            other_symbol = '1' if halting_symbol == '0' else '0'

            # Reading h --> jump to accept
            move_h = random.choice(['L', 'R'])
            transitions[(s, halting_symbol)] = (accept_state, halting_symbol, move_h)

            # Reading o --> cycle.
            next_state = cycle[(i + 1) % len(cycle)]
            move_o = random.choice(['L', 'R'])
            write_o = random.choice(['0', '1'])
            transitions[(s, other_symbol)] = (next_state, write_o, move_o)

        else:
            # never halting: always follow the cycle, writing the opposite of what was read
            for sym in ('0', '1'):
                other_symbol = '1' if sym == '0' else '0'
                next_state = cycle[(i + 1) % len(cycle)]
                move = random.choice(['L', 'R'])
                transitions[(s, sym)] = (next_state, other_symbol, move)

    return transitions
