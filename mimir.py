from graph_injection_coordinator_module import graph_injection_coordinator
from feature_extractor import graph_feature_extractor
from machine_learning_component import train_ml_models

def process_prometheus_data_dump(prometheus_data_dump, timegrans):
    # TODO TODO TODO TODO
    return None

def find_all_microservices(exfil_rate_to_graphs):
    # TODO TODO TODO TODO
    return None

def find_all_exfil_paths(microservices, existing_ACLs):
    # TODO TODO TODO TODO
    return None

def train_the_model(prometheus_data_dump, timegrans, exfil_rates, exfil_time, existing_ACLs):
    timegran_to_graphs = process_prometheus_data_dump(prometheus_data_dump, timegrans)

    microservices = find_all_microservices(timegran_to_graphs)
    possible_exfil_paths = find_all_exfil_paths(microservices, existing_ACLs)

    time_gran_to_injected_graphs = graph_injection_coordinator(timegran_to_graphs, exfil_time, possible_exfil_paths,
                                                               exfil_rates, timegrans)

    time_gran_to_feature_and_label_df = graph_feature_extractor(timegran_to_graphs, time_gran_to_injected_graphs, exfil_rates)

    time_gran_to_ml_model_name_to_Relevanqt_Model_Info = train_ml_models(time_gran_to_feature_and_label_df)

    # TODO: report the performance of the models on the training data (should be v good, since it is the *trainging* data...)

    return  time_gran_to_ml_model_name_to_Relevanqt_Model_Info

def apply_the_existing_models_on_synthetic_data(prometheus_data_dump, timegrans, exfil_rates, exfil_time, existing_ACLs):
    timegran_to_graphs = process_prometheus_data_dump(prometheus_data_dump, timegrans)

    microservices = find_all_microservices(timegran_to_graphs)
    possible_exfil_paths = find_all_exfil_paths(microservices, existing_ACLs)

    time_gran_to_injected_graphs = graph_injection_coordinator(timegran_to_graphs, exfil_time, possible_exfil_paths,
                                                               exfil_rates, timegrans)

    time_gran_to_feature_and_label_df = graph_feature_extractor(timegran_to_graphs, time_gran_to_injected_graphs,
                                                                exfil_rates)

    # TODO: apply the existing models here

    # TODO: report the performance of the models on the training data (should be v good, since it is the *trainging* data...)

    # TODO: what do I want to return ??

def apply_the_existing_models_on_real_data():
    # TODO TODO TODO
    pass

def main():
    # TODO TODO TODO (this'll be more about getting a good interface...)
    pass
