from graph_injection_coordinator_module import graph_injection_coordinator
from feature_extractor import graph_feature_extractor
from machine_learning_component import train_ml_models, run_ml_models, find_model_perform
import pandas as pd
import gen_attack_templates

def process_prometheus_data_dump(prometheus_data_dump, timegrans):
    # TODO TODO TODO TODO
    return None

def find_all_microservices(exfil_rate_to_graphs):
    # TODO TODO TODO TODO
    return None

def find_all_exfil_paths(microservices, existing_ACLs, ms_with_data_to_exfil, max_path_length, dns_porportion,
                         max_number_of_paths):

    netsec_policy, intersvc_vip_pairs = gen_attack_templates.parse_netsec_policy(existing_ACLs)
    synthetic_exfil_paths, initiator_info_for_paths = \
        gen_attack_templates.generate_synthetic_attack_templates(microservices, ms_with_data_to_exfil,
                                                                 max_number_of_paths, existing_ACLs,
                                                                 intersvc_vip_pairs, max_path_length,
                                                                 dns_porportion)
    synthetic_exfil_paths = synthetic_exfil_paths
    initiator_info_for_paths = initiator_info_for_paths
    return synthetic_exfil_paths, initiator_info_for_paths

def train_the_model(prometheus_data_dump, timegrans, exfil_rates, exfil_time, existing_ACLs,
                    ms_with_data_to_exfil, max_path_length, dns_porportion, max_number_of_paths):
    timegran_to_graphs = process_prometheus_data_dump(prometheus_data_dump, timegrans)

    microservices = find_all_microservices(timegran_to_graphs)
    possible_exfil_paths, _ = find_all_exfil_paths(microservices, existing_ACLs, ms_with_data_to_exfil, max_path_length, dns_porportion,
                                                max_number_of_paths)

    time_gran_to_injected_graphs = graph_injection_coordinator(timegran_to_graphs, exfil_time, possible_exfil_paths,
                                                               exfil_rates, timegrans)

    time_gran_to_feature_df, time_gran_to_labels = graph_feature_extractor(timegran_to_graphs, time_gran_to_injected_graphs, exfil_rates,
                                                                is_live=False)

    time_gran_to_ml_model_name_to_Relevanqt_Model_Info = train_ml_models(time_gran_to_feature_df, time_gran_to_labels)

    exfil_paths, exfil_weights = None, None # TODO
    performance_df = generate_performance_info(time_gran_to_ml_model_name_to_Relevanqt_Model_Info, time_gran_to_labels, exfil_paths,
                                               exfil_weights, train=True)

    return time_gran_to_ml_model_name_to_Relevanqt_Model_Info, performance_df

def apply_the_existing_models_on_synthetic_data(prometheus_data_dump, timegrans, exfil_rates, exfil_time, existing_ACLs,
                                                time_gran_to_ml_model_name_to_Relevanqt_Model_Info,
                                                ms_with_data_to_exfil, max_path_length, dns_porportion, max_number_of_paths):

    timegran_to_graphs = process_prometheus_data_dump(prometheus_data_dump, timegrans)

    microservices = find_all_microservices(timegran_to_graphs)
    possible_exfil_paths, _ = find_all_exfil_paths(microservices, existing_ACLs, ms_with_data_to_exfil, max_path_length, dns_porportion,
                                                max_number_of_paths)

    time_gran_to_injected_graphs = graph_injection_coordinator(timegran_to_graphs, exfil_time, possible_exfil_paths,
                                                               exfil_rates, timegrans)

    time_gran_to_feature_df, time_gran_to_labels = graph_feature_extractor(timegran_to_graphs, time_gran_to_injected_graphs,
                                                                exfil_rates, is_live=False)

    timegran_to_ml_model_to_alerts = run_ml_models(time_gran_to_feature_df, time_gran_to_ml_model_name_to_Relevanqt_Model_Info)

    # TODO: report the performance of the models on the training data (should be v good, since it is the *trainging* data...)
    exfil_paths, exfil_weights = None, None # TODO: get these somehow...

    performance_df = generate_performance_info(timegran_to_ml_model_to_alerts, time_gran_to_labels, exfil_paths, exfil_weights)

    # TODO: what do I want to return ??
    return performance_df

def apply_the_existing_models_on_real_data(prometheus_data_dump, timegrans, model_to_use,
                                           time_gran_to_ml_model_name_to_Relevanqt_Model_Info):

    timegran_to_graphs = process_prometheus_data_dump(prometheus_data_dump, timegrans)

    time_gran_to_feature_df, _ = graph_feature_extractor(timegran_to_graphs, None, None, is_live=True)

    ml_model_to_timegran_to_alerts = run_ml_models(time_gran_to_feature_df, time_gran_to_ml_model_name_to_Relevanqt_Model_Info)

    return ml_model_to_timegran_to_alerts[model_to_use]

def generate_performance_info(timegran_to_ml_model_to_alerts, time_gran_to_labels, exfil_paths, exfil_weights, train=False):
    performance_df = pd.DataFrame()
    timegrans = timegran_to_ml_model_to_alerts.keys()
    for timegran in timegrans:
        for ml_model, alerts in timegran_to_ml_model_to_alerts[timegran].iteritems():
            if train:
                alerts = alerts.train_predictions
            current_performance = find_model_perform(alerts, time_gran_to_labels[timegran], exfil_paths, exfil_weights)
            # TODO: add the current performance info to the performance_df (this'll make it easy to look @ later)

    return performance_df

def main():
    # TODO TODO TODO (this'll be more about getting a good interface...)
    # this is probably the next stage TBH...
    pass
