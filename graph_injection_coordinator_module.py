import pandas as pd

def graph_injection_coordinator(time_gran_to_graphs, exfil_time, exfil_paths, exfil_rates, time_grans):
    time_gran_to_injected_graphs  = {}
    for time_gran in time_grans:
        injected_graphs = []
        graphs = time_gran_to_graphs[time_gran]
        exfil_path_index = 0
        graphs_injected_with_current_exfil_path = 0
        cur_injection_info = None # TODO: probably wanna establish a valid class->container mapping here... (or move this statement appropriately)

        for graph in graphs:
            cur_injected_graphs = []
            cur_exfil_path = exfil_paths[exfil_path_index % len(exfil_paths)]

            for exfil_rate in exfil_rates:
                cur_injection_info, injected_graph = inject_graph(graph, exfil_rate, cur_injection_info, cur_exfil_path)
                cur_injected_graphs.append(injected_graph)
            injected_graphs.append(cur_injected_graphs)

            if graphs_injected_with_current_exfil_path >= exfil_time / exfil_time:
                graphs_injected_with_current_exfil_path = 0
                exfil_path_index += 1
                cur_injection_info = None
            else:
                graphs_injected_with_current_exfil_path += 1

        time_gran_to_injected_graphs[time_gran].append(injected_graphs)

    return time_gran_to_injected_graphs

def inject_graph(graph, exfil_rate, cur_injection_info, cur_exfil_path):
    # TODO TODO TODO
    return None, None