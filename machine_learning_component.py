import sklearn
import process_roc
from sklearn.ensemble import AdaBoostRegressor, AdaBoostClassifier, AdaBoostClassifier, AdaBoostRegressor, \
    RandomForestClassifier, ExtraTreesClassifier, RandomTreesEmbedding, BaggingClassifier

class mimir_ml_model:
  def __init__(self, clf, X, Y):
      self.clf = clf
      self.clf.fit(X, Y)
      self.train_predictions = self.clf.predict(X=X)
      _, _, self.optimal_train_thresh, self.optimal_train_f1, self.y_optimal_thresholded = \
          find_optimal_train_threshold(self.train_predictions, Y)
      # TODO need to get: self.exfil_paths_train, self.exfil_weights_train)
      self.train_confusion_matrix = generate_cm(self.y_optimal_thresholded, self.exfil_paths_train, self.exfil_weights_train)
      self.train_performnace = sklearn.metrics.classification_report(Y, self.y_optimal_thresholded, output_dict=True)


def run_ml_models(time_gran_to_feature_df, time_gran_to_ml_model_name_to_Relevanqt_Model_Info):
    time_gran_to_ml_model_name_to_performance = {}
    for time_gran, ml_model_name_to_Relevanqt_Model_Info in time_gran_to_ml_model_name_to_Relevanqt_Model_Info.iteritems():
        ml_model_name_to_performance = {}
        Xt, Yt, exfil_paths_test, exfil_weights_test = None, None, None, None  # TODO
        for ml_model_name, Relevant_Model_Info in ml_model_name_to_Relevanqt_Model_Info.iteritems():
            # TODO TODO TODO :: finish calculating the performance here!
            test_predictions = Relevant_Model_Info.clf.predict(X=Xt)

    # TODO: return alerts.
    return None

def find_model_perform(predicted_values, actual_values, exfil_paths_test, exfil_weights_test):
    attack_type_to_predictions, attack_type_to_truth, attack_type_to_weights = \
        process_roc.determine_categorical_labels(actual_values, predicted_values, exfil_paths_test,
                                                 exfil_weights_test.tolist())

    attack_type_to_confusion_matrix_values = process_roc.determine_cm_vals_for_categories(
        attack_type_to_predictions, attack_type_to_truth)

    # print "attack_type_to_confusion_matrix_values", attack_type_to_confusion_matrix_values

    categorical_cm_df = process_roc.determine_categorical_cm_df(attack_type_to_confusion_matrix_values,
                                                                attack_type_to_weights)
    ## re-name the row without any attacks in it...
    # print "categorical_cm_df.index", categorical_cm_df.index
    confusion_matrix = categorical_cm_df.rename({(): 'No Attack'}, axis='index')

    classification_performance = sklearn.metrics.classification_report(actual_values, predicted_values, output_dict=True)

    return confusion_matrix, classification_performance

def train_ml_models(time_gran_to_feature_df, time_gran_to_labels):
    time_gran_to_ml_model_name_to_Relevanqt_Model_Info = {}

    for time_gran in time_gran_to_feature_df.key():

        current_feature_df = time_gran_to_feature_df[time_gran]
        current_labels = time_gran_to_labels[time_gran]

        ml_model_name_to_Relevanqt_Model_Info = {}

        X, Y, exfil_paths_train, exfil_weights_train = None, None, None, None # TODO

        clf = AdaBoostRegressor(base_estimator=sklearn.linear_model.LassoCV(cv=5, max_iter=80000))
        ml_model_name_to_Relevanqt_Model_Info['AdaBoostRegressor'] = ( mimir_ml_model(clf, X, Y) )

        clf = AdaBoostClassifier(base_estimator=sklearn.linear_model.LogisticRegression())
        ml_model_name_to_Relevanqt_Model_Info['AdaBoostClassifier'] = ( mimir_ml_model(clf, X, Y) )

        clf = AdaBoostRegressor()
        ml_model_name_to_Relevanqt_Model_Info['AdaBoostRegressor_Default'] = ( mimir_ml_model(clf, X, Y) )

        clf = AdaBoostClassifier()
        ml_model_name_to_Relevanqt_Model_Info['AdaBoostClassifier'] = ( mimir_ml_model(clf, X, Y) )

        clf = sklearn.linear_model.LassoCV(cv=5, max_iter=80000)
        ml_model_name_to_Relevanqt_Model_Info['LassoCV'] = ( mimir_ml_model(clf, X, Y) )

        clf = sklearn.linear_model.LogisticRegressionCV()
        ml_model_name_to_Relevanqt_Model_Info['LogisticRegressionCV'] = ( mimir_ml_model(clf, X, Y) )

        clf = RandomForestClassifier()
        ml_model_name_to_Relevanqt_Model_Info['RandomForestClassifier'] = ( mimir_ml_model(clf, X, Y) )

        clf = ExtraTreesClassifier()
        ml_model_name_to_Relevanqt_Model_Info['ExtraTreesClassifier'] = ( mimir_ml_model(clf, X, Y) )

        clf = BaggingClassifier()
        ml_model_name_to_Relevanqt_Model_Info['BaggingClassifier'] = ( mimir_ml_model(clf, X, Y) )

        time_gran_to_ml_model_name_to_Relevanqt_Model_Info[time_gran] = ml_model_name_to_Relevanqt_Model_Info

    return time_gran_to_ml_model_name_to_Relevanqt_Model_Info

def find_optimal_train_threshold(train_predictions, Y):
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(y_true=Y, y_score=train_predictions, pos_label=1)
    list_of_f1_scores = []
    for counter, threshold in enumerate(thresholds):
        # print counter,threshold
        y_pred = [int(i > threshold) for i in train_predictions]
        f1_score = sklearn.metrics.f1_score(Y, y_pred, pos_label=1, average='binary')
        list_of_f1_scores.append(f1_score)
    max_f1_score = max(list_of_f1_scores)
    max_f1_score_threshold_pos = [i for i, j in enumerate(list_of_f1_scores) if j == max_f1_score]
    threshold_corresponding_max_f1 = thresholds[max_f1_score_threshold_pos[0]]
    ########
    train_thresholds = thresholds
    train_f1s = list_of_f1_scores
    optimal_train_thresh = threshold_corresponding_max_f1
    optimal_train_f1 = max_f1_score
    y_optimal_thresholded = [int(i > threshold_corresponding_max_f1) for i in train_predictions]

    return train_thresholds, train_f1s, optimal_train_thresh, optimal_train_f1, y_optimal_thresholded

def generate_cm(y_optimal_thresholded, exfil_paths_train, exfil_weights_train):
    y_train = Y['labels'].tolist()
    # print "y_train", y_train
    attack_type_to_predictions, attack_type_to_truth, attack_type_to_weights = \
        process_roc.determine_categorical_labels(y_train, y_optimal_thresholded, exfil_paths_train,
                                                 exfil_weights_train.tolist())

    attack_type_to_confusion_matrix_values = process_roc.determine_cm_vals_for_categories(attack_type_to_predictions,
                                                                                          attack_type_to_truth)
    categorical_cm_df = process_roc.determine_categorical_cm_df(attack_type_to_confusion_matrix_values,
                                                                attack_type_to_weights)
    ## re-name the row without any attacks in it...
    # print "categorical_cm_df.index", categorical_cm_df.index
    confusion_matrix = categorical_cm_df.rename({(): 'No Attack'}, axis='index')

    return confusion_matrix
