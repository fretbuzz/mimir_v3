import pandas as pd

def graph_feature_extractor(time_gran_to_graphs, time_gran_to_injected_graphs, exfil_rates, is_live):
    time_gran_to_feature_and_label_df = {}
    time_grans = time_gran_to_graphs.keys()

    if is_live:
        for time_gran in time_grans:
            feature_and_label_df = pd.DataFrame()
            legit_graphs = time_gran_to_graphs[time_gran]
            injected_graphs = time_gran_to_injected_graphs[time_gran]

            for graph_index in range(0, len(legit_graphs)):
                cur_injected_graphs = injected_graphs[graph_index]
                cur_legit_graph = legit_graphs[graph_index]
                legit_graph_features = calculate_features_for_current_graph(cur_legit_graph)
                # TODO: append the legit graph features + label to the DF

                for counter, cur_injected_graph in enumerate(cur_injected_graphs):
                    cur_exfil_rate = exfil_rates[counter]
                    malicious_graph_features = calculate_features_for_current_graph(cur_legit_graph)
                    # TODO: append the legit graph features + label to the DF
    else:
        for time_gran in time_grans:
            possibily_malicious_graphs = time_gran_to_graphs[time_gran]

            for graph_index in range(0, len(possibily_malicious_graphs)):
                cur_possibily_malicious_graphs = possibily_malicious_graphs[graph_index]
                legit_graph_features = calculate_features_for_current_graph(cur_possibily_malicious_graphs)
                # TODO: append the possibly malicious graph features + label (which is '?' here) to the DF

    return time_gran_to_feature_df, time_gran_to_labels

def calculate_features_for_current_graph(graph):
    # TODO TODO TODO
    return None


