from collections import defaultdict
from math import pow

def create_map_truth_table(TT, x, depth, f):
    if depth == len(x):
        TT[tuple(x)] = f[len(TT)]
    else:
        x[depth] = 1
        create_map_truth_table(TT, x, depth + 1, f)
        x[depth] = 0
        create_map_truth_table(TT, x, depth + 1, f)

def complementary_subset(subset, N):
    return [i for i in range(N) if i not in subset]

def calculate_g_functions(x, S, pos_s, B, g, g_set, N, TT):
    if pos_s < 0:
        if B:
            S_c = complementary_subset(S, N)
            g = []
            calculate_g_functions(x, S_c, len(S_c) - 1, False, g, g_set, N, TT)
            g_set.add(tuple(g))
        else:
            g.append(TT[tuple(x)])
    else:
        x[S[pos_s]] = 1
        calculate_g_functions(x, S, pos_s - 1, B, g, g_set, N, TT)
        x[S[pos_s]] = 0
        calculate_g_functions(x, S, pos_s - 1, B, g, g_set, N, TT)

def calculate_information_shared(S, N, TT):
    S_c = complementary_subset(S, N)
    x = [0] * N
    fun_set = set()
    calculate_g_functions(x, S, len(S) - 1, True, [], fun_set, N, TT)
    i = len(fun_set)
    fun_set2 = set()
    calculate_g_functions(x, S_c, len(S_c) - 1, True, [], fun_set2, N, TT)
    i_c = len(fun_set2)
    return i + i_c

def entanglement_recursive(subset, C, index, N, TT):
    if len(subset) != N // 2:
        ent = float('inf')
    else:
        ent = calculate_information_shared(subset, N, TT)
    for i in range(index, N):
        subset.append(C[i])
        ent = min(ent, entanglement_recursive(subset, C, i + 1, N, TT))
        subset.pop()
    return ent

def entanglement(f, N):
    TT = {}
    x = [0] * N
    create_map_truth_table(TT, x, 0, f)
    C = list(range(N))
    return entanglement_recursive([], C, 0, N, TT)
