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

def generate_binary_counter_transitions():
    """
    Only works for machines with 2 bits for state (4 states total)
    """
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

def generate_alternating_counter_transitions():
    """
    Transition table for the 8-state Turing machine that

      • starts with all-zero bits between L* and R*
      • repeatedly increments the binary counter
      • halts ⇔  the bits (read MSB→LSB, i.e. from R* towards L*) form
                 the alternating pattern 1 0 1 0 …  (length arbitrary)

    Alphabet  :  {'L*', 'R*', '0', '1'}
    Directions:  'L' = left,  'R' = right
    States     : 0–7 (no explicit accept state; halting = accepting)

    Tuple order:
        transition_function[(state, symbol)] = (next_state, write_symbol, direction)
    """
    tf = {}

    # --- 0 : start / generic rebobinado (head over L*) ------------------------
    tf[(0, 'L*')] = (1, 'L*', 'R')           # jump to LSB
    for s in ('0', '1', 'R*'):
        tf[(0, s)] = (0, s, 'L')             # keep running left until L*

    # --- 1 : increment (check only the LSB) ----------------------------------
    tf[(1, '0')] = (3, '1', 'R')             # 0 → 1   (no carry)
    tf[(1, '1')] = (2, '0', 'R')             # start carry

    # --- 2 : propagate carry --------------------------------------------------
    tf[(2, '1')]  = (2, '0', 'R')            # keep propagating
    tf[(2, '0')]  = (3, '1', 'R')            # first 0 → 1, carry resolved
    tf[(2, 'R*')] = (7, 'R*', 'L')           # overflow: adjust, then verify

    # --- 7 : repair after overflow (write new MSB = 1) ------------------------
    tf[(7, '0')] = (3, '1', 'R')
    tf[(7, '1')] = (3, '1', 'R')             # (already 1) just continue

    # --- 3 : run right until we see R* ---------------------------------------
    for s in ('0', '1'):
        tf[(3, s)] = (3, s, 'R')
    tf[(3, 'R*')] = (4, 'R*', 'L')           # step onto MSB and begin check

    # --- 4 : verifier – expect 1 ---------------------------------------------
    tf[(4, '1')] = (5, '1', 'L')             # correct, now expect 0
    tf[(4, '0')] = (6, '0', 'L')             # mismatch: rewind
    tf[(4, 'R*')] = (6, 'R*', 'L')           # impossible but safe route
    # 4 + L*  →  HALT (length even, ends in 0)  → accept

    # --- 5 : verifier – expect 0 ---------------------------------------------
    tf[(5, '0')] = (4, '0', 'L')             # correct, now expect 1
    tf[(5, '1')] = (6, '1', 'L')             # mismatch: rewind
    tf[(5, 'R*')] = (6, 'R*', 'L')
    # 5 + L*  →  HALT (length odd, ends in 1)  → accept

    # --- 6 : fast rewind after verification failure --------------------------
    for s in ('0', '1', 'R*'):
        tf[(6, s)] = (6, s, 'L')             # keep running left
    tf[(6, 'L*')] = (1, 'L*', 'R')           # restart at LSB

    return tf


def counter_max_steps(tape_bits: int, alternating = False):
    """
    Upeer bound (or exact value) of steps that BinaryCounter gives.
    """
    t = tape_bits
    t_2 = 1 << t
    
    if alternating: 
        return 8 / 3 * t_2 * (t + 1)
    else: 
        return 4 * t_2 - t - 3
