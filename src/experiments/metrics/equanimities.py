from collections import defaultdict
from math import factorial, pow, log2

def get_N(f):
    size = len(f)
    if size == 0 or (size & (size - 1)) != 0:
        raise ValueError("Length of f must be a power of two")
    return int(log2(size))

def number_of_subsets(k: int, N: int) -> int:
    prod = 1
    for i in range(k):
        prod *= 2 * (N - i)
    return prod // factorial(k)

def update_counters(count, subset, x, index):
    count[len(subset)][tuple(subset)] += 1
    for i in range(index, len(x)):
        subset.append(x[i])
        update_counters(count, subset, x, i + 1)
        subset.pop()

def truth_table(x, index, f, pos, count, N):
    if index < 0:
        if f[pos[0]] == 1:
            subset = []
            update_counters(count, subset, x, 0)
        pos[0] += 1
    else:
        x[N - 1 - index] = index
        truth_table(x, index - 1, f, pos, count, N)
        x[N - 1 - index] = index + N
        truth_table(x, index - 1, f, pos, count, N)

def equanimity_subsets_normalized(f):
    
    N = get_N(f)
    
    x = [0] * N
    v_count = [defaultdict(int) for _ in range(N + 1)]
    pos = [0]
    truth_table(x, N - 1, f, pos, v_count, N)
    eq = 0
    for k in range(1, N + 1):
        sum_values = sum(v_count[k].values())
        num_subsets = number_of_subsets(k, N)
        avg = sum_values / num_subsets
        variance = sum((val - avg) ** 2 for val in v_count[k].values())
        variance += (num_subsets - len(v_count[k])) * (avg ** 2)
        variance /= num_subsets
        eq += variance / pow(2, 2 * (N - k - 1))
    return 1 - eq / N

def equanimity_subsets(f):
    
    N = get_N(f)
    
    x = [0] * N
    v_count = [defaultdict(int) for _ in range(N + 1)]
    pos = [0]
    truth_table(x, N - 1, f, pos, v_count, N)
    eq = 0
    for k in range(1, N + 1):
        num_subsets = number_of_subsets(k, N)
        sum_values = sum(v_count[k].values())
        avg = sum_values / num_subsets
        variance = sum((val - avg) ** 2 for val in v_count[k].values())
        variance += (num_subsets - len(v_count[k])) * (avg ** 2)
        variance /= num_subsets
        eq += variance
    return -eq

def equanimity_importance(f):
    
    N = get_N(f)
    
    I = 0
    for i in range(1, N + 1):
        for j in range(0, int(pow(2, N)), int(pow(2, i))):
            for k in range(int(pow(2, i - 1))):
                if f[k + j] != f[k + j + int(pow(2, i - 1))]:
                    I += 1
    return I / (N * pow(2, N - 1))
