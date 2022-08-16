# Application script for automated River

if __name__ == '__main__':

    # imports

    import numpy as np
    import pandas as pd
    import arff

    from river import metrics
    from river.drift import EDDM
    from river import neighbors
    from river import ensemble
    from river import preprocessing
    from river import linear_model
    from river import tree
    from river import naive_bayes
    from river import evaluate
    from river import datasets
    from river import stream
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import LabelEncoder
    from gama.utilities.river_stream_stats import StreamStats

    from skmultiflow import drift_detection
    from skmultiflow.drift_detection import EDDM
    from skmultiflow.drift_detection.adwin import ADWIN
    import wandb

    from data_streams import classification_datasets, regression_datasets, load_format_data, cls_num, reg_num

    import sys
    import os

    # Models

    model_1 = tree.ExtremelyFastDecisionTreeClassifier()
    model_2 = preprocessing.StandardScaler() | linear_model.Perceptron()
    model_3 = preprocessing.AdaptiveStandardScaler() | tree.HoeffdingAdaptiveTreeClassifier()
    model_4 = tree.HoeffdingAdaptiveTreeClassifier()
    model_5 = ensemble.LeveragingBaggingClassifier(
        preprocessing.StandardScaler() | linear_model.Perceptron())
    model_6 = preprocessing.StandardScaler() | neighbors.KNNClassifier()
    model_7 = naive_bayes.BernoulliNB()

    model_8 = preprocessing.StandardScaler() | ensemble.AdaptiveRandomForestRegressor()
    model_9 = preprocessing.AdaptiveStandardScaler() | tree.HoeffdingAdaptiveTreeRegressor()
    model_10 = preprocessing.AdaptiveStandardScaler() | ensemble.BaggingRegressor(
        linear_model.LinearRegression())

    # User parameters

    print(sys.argv[0])  # prints python_script.py
    # prints dataset no
    print(f"Data stream is {sys.argv[1]}.")
    # prints initial batch size
    print(f"Initial batch size is {int(sys.argv[2])}.")

    data_loc = sys.argv[1]  # needs to be arff
    initial_batch = int(sys.argv[2])  # initial set of samples to train automl
    live_plot = True

    # verify dataset is available

    assert data_loc in regression_datasets + \
        classification_datasets, f"Path to dataset '{data_loc}' not found"

    # set task for given dataset
    if data_loc in regression_datasets:
        task = "regression"
        WANDB_PROJECT = "Results"
        model = model_8
        is_classification = False
        online_metric = metrics.RMSE()  # river metric to evaluate online learning

    elif data_loc in classification_datasets:
        task = "classification"
        WANDB_PROJECT = "OAML Classification"
        model = model_5
        is_classification = True
        online_metric = metrics.Accuracy()  # river metric to evaluate online learning

    # logging info
    # strategy = sys.argv[0].split("_")[2].split(".")[0]
    data_num = cls_num[data_loc] if is_classification else reg_num[data_loc]
    filename = f"results_extended_LB_{data_num}" if is_classification else f"results_extended_LB_{data_num}"

    path_name = f"oaml_paper/classification_results/LB/{filename}" if is_classification else f"oaml_paper/classification_results/LB/{filename}"
    open(f"{path_name}.txt", "w+")

    # EDDM only compatible with classification
    drift_detector = EDDM() if is_classification else ADWIN()

    # Plot initialization
    if live_plot:
        os.environ["WANDB_START_METHOD"] = "thread"
        wandb.init(
            name="Leverage Bagging",
            project=WANDB_PROJECT,
            entity="lota",
            tags=["REGRESSION", "baseline"],
            config={
                "dataset": data_loc,
                "online_performance_metric": online_metric,
            })

    # Data
    X, y = load_format_data(data_loc, task, target_last=True)
    logging_point = int(len(X) * 0.01)
    online_stats = StreamStats(initial_batch)

    # initial training
    if not is_classification:
        print(f'Test batch - 0 with RMSE: 0', file=open(f"{path_name}.txt", "a"))
    else:
        print(f'Test batch - 0 with Accuracy: 0', file=open(f"{path_name}.txt", "a"))
    for i in range(0, initial_batch):
        model = model.learn_one(X.iloc[i].to_dict(), y[i])

    for i in range(initial_batch + 1, len(X)):

        if not is_classification:
            y[i] = online_stats.validate_y(y[i])

        # Test then train - by one
        y_pred = model.predict_one(X.iloc[i].to_dict())
        online_metric = online_metric.update(y[i], y_pred)
        model = model.learn_one(X.iloc[i].to_dict(), y[i])

        # Print performance every x interval
        if i % logging_point == 0:
            print(f'Test batch - {i} with {online_metric}',
                  file=open(f"{path_name}.txt", "a"))
            if live_plot:
                wandb.log(
                    {"current_point": i, "Prequential performance": online_metric.get()})

        # #Check for drift
        # #in_drift, in_warning = drift_detector.update(int(y_pred == y[i]))
        # drift_detector.add_element(int(y_pred != y[i]))
        # #if in_drift:
        # if drift_detector.detected_change():
        #     print(f"Change detected at data point {i} and current performance is at {online_metric}")
        #     if live_plot:
        #         wandb.log({"drift_point": i, "current_point": i, "Prequential performance": online_metric.get()})
