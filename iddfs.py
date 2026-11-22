# GRIDMIND - Fault Diagnosis using IDDFS

import time

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

start_node = 'S1'

transformers = ['T1', 'T2', 'T3', 'T4']

# Four testing scenarios
scenarios = {
    'Scenario 1': 'T1',
    'Scenario 2': 'T2',
    'Scenario 3': 'T3',
    'Scenario 4': 'T4'
}


# Generate observations for each scenario
def create_observations(fault):

    observations = {}

    for node in network:
        observations[node] = 'NORMAL'

    observations[fault] = 'ABNORMAL'

    # Consumer areas connected to faulty transformer
    for node in network[fault]:
        if node.startswith('L'):
            observations[node] = 'ABNORMAL'

    return observations


# Depth Limited Search
def depth_limited_search(node, parent, depth, limit,
                         path, reached, faults, affected,
                         stats, observations):

    stats['nodes'] += 1
    reached.add(node)

    # Fault diagnosis rule:
    # An abnormal transformer with a normal parent
    # is considered the root cause.
    if observations[node] == 'ABNORMAL':

        if parent is None or observations[parent] == 'NORMAL':

            if node in transformers:
                faults.add(node)

        elif node.startswith('L'):
            affected.add(node)

    # Check depth cutoff
    if depth == limit:

        cutoff = False

        for neighbour in network[node]:

            if neighbour in path:
                stats['pruned'] += 1
            else:
                cutoff = True

        return cutoff

    cutoff = False

    # Explore neighbouring nodes
    for neighbour in network[node]:

        if neighbour not in path:

            result = depth_limited_search(
                neighbour, node, depth + 1, limit,
                path + [neighbour], reached,
                faults, affected, stats, observations
            )

            if result:
                cutoff = True

        else:
            # Skip cycles
            stats['pruned'] += 1

    return cutoff


# Iterative Deepening DFS
def iddfs(start, observations):

    max_depth = len(network)

    stats = {
        'nodes': 0,
        'pruned': 0
    }

    faults = set()
    affected = set()
    reached = set()

    for limit in range(max_depth + 1):

        reached = set()
        faults = set()
        affected = set()

        cutoff = depth_limited_search(
            start, None, 0, limit,
            [start], reached, faults,
            affected, stats, observations
        )

        # Stop when no depth cutoff remains
        if not cutoff:
            break

    return faults, affected, reached, stats


# Test all four scenarios
if __name__ == '__main__':

    for scenario_name, actual_fault in scenarios.items():

        print("\nTesting:", scenario_name)
        print("Starting Node:", start_node)

        observations = create_observations(actual_fault)

        start_time = time.perf_counter()

        # Run IDDFS
        faults, affected, reached, stats = iddfs(
            start_node, observations
        )

        end_time = time.perf_counter()

        # Determine inspection order
        inspection_order = []

        for transformer in transformers:

            inspection_order.append(transformer)

            if transformer in faults:
                break

        # Display inspection results
        for transformer in inspection_order:

            if transformer in faults:
                print(f"Inspect {transformer} : FAULT")
            else:
                print(f"Inspect {transformer} : NORMAL")

        print("Inspection Order:", inspection_order)

        # Independently identify the scenario
        identified_scenario = "No fault identified"

        for name, fault in scenarios.items():

            if fault in faults:
                identified_scenario = name
                break

        print("Actual Scenario:", scenario_name)
        print("Identified Scenario:", identified_scenario)

        print("Detected Faults:", sorted(faults))

        print("Nodes Explored:", stats['nodes'])

        print("Cycle Edges Skipped:", stats['pruned'])

        execution_time = (end_time - start_time) * 1000

        print(f"Execution Time: {execution_time:.4f} ms")

        print("Affected Areas:", sorted(affected))
