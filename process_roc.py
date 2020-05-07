import pandas as pd
'''from old project...'''

def determine_categorical_labels(y_test, optimal_predictions, exfil_paths, exfil_weights):
    attack_type_to_predictions = {}
    attack_type_to_truth = {}
    print exfil_paths
    print exfil_paths.tolist()
    types_of_exfil_paths = []
    for gen_exfil_path in exfil_paths.tolist():
        if str(gen_exfil_path) not in types_of_exfil_paths:
            types_of_exfil_paths.append(str(gen_exfil_path))
    #types_of_exfil_paths = list(set(exfil_paths.tolist()))
    print "types_of_exfil_paths", types_of_exfil_paths
    types_of_exfil_paths_eval = []
    for i in types_of_exfil_paths:
        print("i",i)
        types_of_exfil_paths_eval.append(ast.literal_eval(i))
    types_of_exfil_paths = types_of_exfil_paths_eval
    #types_of_exfil_paths = [ast.literal_eval(i) for i in types_of_exfil_paths]
    print "types_of_exfil_paths", types_of_exfil_paths
    types_of_exfil_paths = [i if i != 0 else [] for i in types_of_exfil_paths]
    print "types_of_exfil_paths", types_of_exfil_paths
    attack_type_to_index = {}
    for exfil_type in types_of_exfil_paths:
        print "exfil_type",exfil_type, type(exfil_type), "tuple(exfil_type)",tuple(exfil_type), type(tuple(exfil_type))
        current_indexes = [i for i, j in enumerate(exfil_paths) if str(j) == str(exfil_type)]
        print "current_indexes", current_indexes
        attack_type_to_index[tuple(exfil_type)] = current_indexes
    number_of_found_exfils = len(attack_type_to_index.values())
    number_of_existing_times = len(y_test)
    if number_of_found_exfils == number_of_existing_times:
        print "number_of_found_exfils == number_of_existing_times", number_of_found_exfils, number_of_existing_times
        #traceback.print_exc(file=sys.stdout)
        exit(312)
    print "optimal_predictions", len(optimal_predictions)
    print "attack_type_to_index", len(attack_type_to_index)
    #print "y_test", y_test['labels'], type(y_test['labels'])
    attack_type_to_weights = {}
    for exfil_type,indexes in attack_type_to_index.iteritems():
        print len(indexes), len(optimal_predictions), indexes
        attack_type_to_predictions[exfil_type] = [optimal_predictions[i] for i in indexes]
        attack_type_to_truth[exfil_type] = [y_test[i] for i in indexes]
        print len(y_test), type(y_test), y_test
        print len(exfil_weights), type(exfil_weights), exfil_weights
        attack_type_to_weights[exfil_type] = [exfil_weights[i] for i in indexes]
    return attack_type_to_predictions, attack_type_to_truth, attack_type_to_weights

def determine_cm_vals_for_categories(attack_type_to_predictions, attack_type_to_truth):
    attack_type_to_confusion_matrix_values = {}
    for attack_type, predictions in attack_type_to_predictions.iteritems():
        truth = attack_type_to_truth[attack_type]
        print "attack_type", attack_type
        print "truth", truth
        print "predictions", predictions
        if truth != [] and predictions != []:
            print "truth_and_predictions_not_empty"
            print sklearn.metrics.confusion_matrix(truth, predictions,labels=[0,1]).ravel()
            tn, fp, fn, tp = sklearn.metrics.confusion_matrix(truth, predictions, labels=[0,1]).ravel()

            ### FIX THIS AT SOME POINT... since
            # for no attack, negatives are really positive and vice-versa...
            ##### UPDATE: NO!!! This made sense at the time b/c I was trying to shoe-horn in f1's for graphs, but
            ##### it does NOT actually make sense!!!
            #if 'No Attack' in attack_type or attack_type == ():
            #    tn,fn,tp,fp = tp,fp,tn,fn

        else:
            print "truth_and_predictions_empty"
            tn, fp, fn, tp = 0, 0, 0, 0
        attack_type_to_confusion_matrix_values[attack_type] = {}
        attack_type_to_confusion_matrix_values[attack_type]['tn'] = tn
        attack_type_to_confusion_matrix_values[attack_type]['tp'] = tp
        attack_type_to_confusion_matrix_values[attack_type]['fp'] = fp
        attack_type_to_confusion_matrix_values[attack_type]['fn'] = fn
    return attack_type_to_confusion_matrix_values


def determine_categorical_cm_df(attack_type_to_confusion_matrix_values, attack_type_to_weights):
    print "attack_type_to_confusion_matrix_values", attack_type_to_confusion_matrix_values
    index = attack_type_to_confusion_matrix_values.keys()
    columns = attack_type_to_confusion_matrix_values[index[0]].keys() + ['exfil_weights']
    categorical_cm_df = pd.DataFrame(0, index=index, columns=columns, dtype='object')
    for attack_type, confusion_matrix_values in attack_type_to_confusion_matrix_values.iteritems():
        for cm_value_types, cm_values in confusion_matrix_values.iteritems():
            print "attack_type", attack_type
            print "df_indexes", categorical_cm_df.index
            print attack_type in categorical_cm_df.index
            print categorical_cm_df[cm_value_types]
            categorical_cm_df[cm_value_types][attack_type] = cm_values

        print attack_type_to_weights[attack_type]
        categorical_cm_df['exfil_weights'][attack_type] = attack_type_to_weights[attack_type]
    return categorical_cm_df