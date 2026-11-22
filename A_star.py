import heapq
import time

network = {
    'G1': ['S1'], 'G2': ['S3'],
    'S1': ['G1', 'T1', 'T2', 'S3', 'S2'], 'S2': ['S1', 'T3'], 'S3': ['G2', 'S1', 'T4'],
    'T1': ['S1', 'L1', 'L2'], 'T2': ['S1', 'L3', 'L4'], 'T3': ['S2', 'L5', 'L6'], 'T4': ['S3', 'L7', 'L8'],
    'L1': ['T1'], 'L2': ['T1'], 'L3': ['T2'], 'L4': ['T2'],
    'L5': ['T3'], 'L6': ['T3'], 'L7': ['T4'], 'L8': ['T4']
}
start_node = 'S1'
transformers = ['T1', 'T2', 'T3', 'T4']
node_layer = {'G': 0, 'S': 1, 'T': 2, 'L': 3}

scenarios = {
    "Scenario 1": {'T1': 'FAULT',  'T2': 'NORMAL', 'T3': 'NORMAL', 'T4': 'NORMAL'},
    "Scenario 2": {'T1': 'NORMAL', 'T2': 'FAULT',  'T3': 'NORMAL', 'T4': 'NORMAL'},
    "Scenario 3": {'T1': 'NORMAL', 'T2': 'NORMAL', 'T3': 'FAULT',  'T4': 'NORMAL'},
    "Scenario 4": {'T1': 'NORMAL', 'T2': 'NORMAL', 'T3': 'NORMAL', 'T4': 'FAULT'}
}

def astar_shortest_path(graph, start, goal):
    """A* pathfinding with an admissible layer-difference heuristic."""
    def heuristic(node):
        if node == goal:
            return 0
        return max(1, abs(node_layer[node[0]] - node_layer[goal[0]]))

    open_list = [(heuristic(start), 0, start, [start])]
    closed_list = set()
    nodes_expanded = 0

    while open_list:
        f_cost, g_cost, current, path = heapq.heappop(open_list)
        if current in closed_list:
            continue
        closed_list.add(current)
        nodes_expanded += 1

        if current == goal:
            return path, g_cost, nodes_expanded

        for neighbor in graph[current]:
            if neighbor not in closed_list:
                new_g = g_cost + 1
                new_f = new_g + heuristic(neighbor)
                heapq.heappush(open_list, (new_f, new_g, neighbor, path + [neighbor]))

    return None, float('inf'), nodes_expanded

def diagnose_grid_astar(graph, start, transformer_list, actual_states, scenario_name):
    start_time = time.perf_counter()

    current_node = start
    uninspected = list(transformer_list)
    inspection_log = []
    detected_faults = []
    total_nodes_explored = 0
    total_path_cost = 0

    while uninspected:
        # Logical deduction: if 3 transformers are NORMAL, the last one must be FAULT
        if len(uninspected) == 1 and not detected_faults:
            detected_faults.append(uninspected[0])
            break

        candidates = []
        for idx, target in enumerate(uninspected):
            path, g_cost, explored = astar_shortest_path(graph, current_node, target)
            total_nodes_explored += explored
            h_cost = 0 if len(uninspected) == 1 else 1
            candidates.append((g_cost + h_cost, g_cost, idx, target, path))

        candidates.sort()
        _, best_g, _, next_target, _ = candidates[0]

        current_node = next_target
        total_path_cost += best_g
        uninspected.remove(next_target)

        status = actual_states[next_target]
        inspection_log.append((next_target, status))

        if status == 'FAULT':
            detected_faults.append(next_target)
            break

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # Identify scenario independently from detected faults
    identified = next(
        (s for s, st in scenarios.items() if [t for t, v in st.items() if v == 'FAULT'] == detected_faults),
        "Unknown"
    )
    affected = sorted(n for f in detected_faults for n in graph[f] if n.startswith('L'))

    print(f"Testing: {scenario_name}")
    print(f"Starting Node: {start}")
    for t, st in inspection_log:
        print(f"Inspect {t} : {st}")
    print(f"Inspection Order: {[t for t, _ in inspection_log]}")
    print(f"Identified Scenario: {identified}")
    print(f"Detected Faults: {detected_faults}")
    print(f"Affected Areas: {affected}")
    print(f"Total Path Cost g(n): {total_path_cost}")
    print(f"Nodes Explored: {total_nodes_explored}")
    print(f"Execution Time: {elapsed_ms:.4f} ms\n")

if __name__ == "__main__":
    for name, states in scenarios.items():
        diagnose_grid_astar(network, start_node, transformers, states, name)