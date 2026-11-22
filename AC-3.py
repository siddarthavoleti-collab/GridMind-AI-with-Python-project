import time
from collections import deque

network = {'G1':['S1'],'G2':['S3'],'S1':['G1','T1','T2','S3','S2'],'S2':['S1','T3'],'S3':['G2','S1','T4'],
 'T1':['S1','L1','L2'],'T2':['S1','L3','L4'],'T3':['S2','L5','L6'],'T4':['S3','L7','L8'],
 'L1':['T1'],'L2':['T1'],'L3':['T2'],'L4':['T2'],'L5':['T3'],'L6':['T3'],'L7':['T4'],'L8':['T4']}
start_node = 'S1'

# Power flows G -> S -> T -> L ; break the S1<->S3 cycle (S1 fed by G1, S3 by G2)
rank = lambda n: {'G': 0, 'S': 1, 'T': 2, 'L': 3}[n[0]]
parents = {n: [m for m in network[n] if rank(m) < rank(n)] for n in network}
parents['S1'] = ['G1']; parents['S3'] = ['G2']; parents['S2'] = ['S1']   # S2 fed from S1

# Scenarios: hidden ground truth (a fault in T makes T and its loads ABNORMAL)
scenarios = {'Scenario 1': ['T1'], 'Scenario 2': ['T2'],
             'Scenario 3': ['T3'], 'Scenario 4': ['T4']}

def make_observations(faults):
    obs = {n: 'NORMAL' for n in network}
    for f in faults:
        obs[f] = 'ABNORMAL'
        for child in network[f]:
            if rank(child) > rank(f):
                obs[child] = 'ABNORMAL'
    return obs

def ac3_diagnose(obs):
    domains = {n: {0, 1} for n in network}          # 1 = faulty
    for n, o in obs.items():                         # node consistency
        if o == 'NORMAL':
            domains[n] = {0}
    constraints = {}
    def add(x, y, rel):
        constraints[(x, y)] = rel
        constraints[(y, x)] = {(b, a) for a, b in rel}
    for n, o in obs.items():                         # ABNORMAL x => F[x] or F[parent]
        if o == 'ABNORMAL':
            for p in parents[n]:
                add(n, p, {(a, b) for a in (0, 1) for b in (0, 1) if a or b})

    explored = pruned = 0
    q = deque(constraints)
    while q:
        X, Y = q.popleft()
        explored += 1
        removed = False
        for x in set(domains[X]):
            if not any((x, y) in constraints[(X, Y)] for y in domains[Y]):
                domains[X].discard(x); pruned += 1; removed = True
        if removed:
            if not domains[X]:
                return None, explored, pruned
            q.extend((Z, X) for (Z, W) in constraints if W == X and Z != Y)

    def consistent(a):
        return all((a[x], a[y]) in r for (x, y), r in constraints.items() if x in a and y in a)
    def backtrack(a, left):
        if not left: return a
        v, rest = left[0], left[1:]
        for val in sorted(domains[v]):               # try 0 first -> fewest faults
            a[v] = val
            if consistent(a):
                res = backtrack(a, rest)
                if res: return res
            del a[v]
        return None

    assign = {n: min(d) for n, d in domains.items() if len(d) == 1}
    sol = backtrack(assign, [n for n, d in domains.items() if len(d) > 1])
    return sorted(n for n, v in sol.items() if v), explored, pruned

def inspect_order(obs):
    """Inspect transformers in order; stop at the first FAULT. The last one is inferred."""
    order, ts = [], ['T1', 'T2', 'T3', 'T4']
    for t in ts[:-1]:
        order.append((t, obs[t]))
        if obs[t] == 'ABNORMAL':
            break
    return order

for name, faults in scenarios.items():
    print(f"Testing: {name}")
    print(f"Starting Node: {start_node}")
    obs = make_observations(faults)

    t0 = time.perf_counter()
    order = inspect_order(obs)
    for t, o in order:
        print(f"Inspect {t} : {'FAULT' if o == 'ABNORMAL' else 'NORMAL'}")
    print(f"Inspection Order: {[t for t, _ in order]}")
    detected, explored, pruned = ac3_diagnose(obs)
    ms = (time.perf_counter() - t0) * 1000

    identified = next((s for s, f in scenarios.items() if f == detected), 'Unknown')
    print(f"Identified Scenario: {identified}")
    print(f"Detected Faults: {detected}")
    print(f"Nodes Explored: {explored}")
    print(f"Branches Pruned: {pruned}")
    print(f"Execution Time: {round(ms, 4)} ms\n")