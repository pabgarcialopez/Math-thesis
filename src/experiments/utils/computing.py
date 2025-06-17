from pyeda.inter import exprvars, truthtable # type: ignore
from pyeda.boolalg.minimization import espresso_tts # type: ignore
import numpy as np

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

def bucket(vals, bounds):
    """
    Given a list of numbers `vals` and two thresholds (lo, hi) in `bounds`,
    returns counts of [ <=lo, (lo, hi], >hi ].
    """
    lo, hi = bounds
    short = sum(1 for v in vals if v <= lo)
    med   = sum(1 for v in vals if lo < v <= hi)
    long  = len(vals) - short - med
    return short, med, long