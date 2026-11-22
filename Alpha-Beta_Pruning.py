#CS25B2029
#K.VAISHNAV
#ALPHA-BETA 
import time

# 1. GRIDMIND NETWORK

network = {
    'G1': ['S1'],
    'G2': ['S3'],

    'S1': ['G1', 'T1', 'T2', 'S3', 'S2'],
    'S2': ['S1', 'T3'],
    'S3': ['G2', 'S1', 'T4'],

    'T1': ['S1', 'L1', 'L2'],
    'T2': ['S1', 'L3', 'L4'],
    'T3': ['S2', 'L5', 'L6'],
    'T4': ['S3', 'L7', 'L8'],

    'L1': ['T1'],
    'L2': ['T1'],
    'L3': ['T2'],
    'L4': ['T2'],
    'L5': ['T3'],
    'L6': ['T3'],
    'L7': ['T4'],
    'L8': ['T4']
}


# Get transformer names from the network
transformers = [node for node in network if node.startswith('T')]


# 2. FAULT SCENARIOS

faults = {
    "Scenario 1": ["T1"],
    "Scenario 2": ["T2"],
    "Scenario 3": ["T3"],
    "Scenario 4": ["T4"],
}


# 3. COUNTERS

nodes = 0
pruned = 0

# 4. ALPHA-BETA PRUNING

#Algorithm
#Adversial Search
#Alpha Beta Pruning

def alpha_beta(candidates, remaining, depth, alpha, beta):

    #Declared as Global
    global nodes, pruned
    nodes += 1

    if len(candidates) == 1:
        return -depth, None

    if len(remaining) == 0:
        return -1000, None

    #-infinity
    best = float('-inf')
    choice = None

    # MAX chooses a transformer
    for i, t in enumerate(remaining):

        fault_group = [s for s in candidates if t in faults[s]]
        normal_group = [s for s in candidates if t not in faults[s]]

        groups = [g for g in [fault_group, normal_group] if g]

        # MIN considers the worst outcome
        worst = float('inf')

        for j, group in enumerate(groups):

            score, _ = alpha_beta(group,remaining[:i] + remaining[i+1:],depth + 1,alpha,min(beta, worst))

            worst = min(worst, score)

            # Alpha-Beta pruning
            if worst <= alpha:
                pruned += len(groups) - j - 1
                break

        if worst > best:
            best = worst
            choice = t

        alpha = max(alpha, best)

        #Pruning Condition
        if alpha >= beta:
            pruned += len(remaining) - i - 1
            break

    return best, choice


# 5. TEST EACH FAULT SCENARIO

def run_test(name, actual_faults):

    global nodes, pruned

    nodes = 0
    pruned = 0

    candidates = list(faults.keys())
    remaining = transformers.copy()
    order = []

    #Time Calculation
    start = time.perf_counter()

    print("\nTesting:", name)

    while len(candidates) > 1 and remaining:

        score, chosen = alpha_beta(candidates,remaining,0,float('-inf'),float('inf'))

        if chosen is None:
            break

        order.append(chosen)
        remaining.remove(chosen)

        # Simulate transformer inspection
        if chosen in actual_faults:
            result = "FAULT"
        else:
            result = "NORMAL"

        print("Inspect", chosen, ":", result)

        # Update possible scenarios
        candidates = [
            s for s in candidates
            if (chosen in faults[s]) == (result == "FAULT")
        ]

    end = time.perf_counter()

    print("Inspection Order:", order)

    if len(candidates) == 1:
        print("Identified Scenario:", candidates[0])
        print("Detected Faults:", faults[candidates[0]])
    else:
        print("Scenario not identified")

    print("Recursive Calls:", nodes)
    print("Branches Skipped:", pruned)
    print("Execution Time:",round((end - start) * 1000, 4), "ms")


# 6. RUN ALL SCENARIOS

print("GRIDMIND - ALPHA BETA PRUNING")

for name, actual in faults.items():
    run_test(name, actual)
