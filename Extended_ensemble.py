# Application script for automated River

if __name__ == '__main__':

    # imports
    import numpy as np
    import pandas as pd
    import arff
    import sys
    import os

    import warnings
    warnings.filterwarnings("ignore")

    from gama import GamaClassifier, GamaRegressor
    from gama.search_methods import AsyncEA
    from gama.search_methods import RandomSearch
    from gama.search_methods import AsynchronousSuccessiveHalving
    from gama.postprocessing import BestFitOnlinePostProcessing
    from gama.utilities.river_comparison import replace_current_model, select_ensemble_from
    from gama.utilities.river_stream_stats import StreamStats

    from river import metrics
    from river import evaluate
    from river import stream
    from river import ensemble

    from skmultiflow import drift_detection
    from skmultiflow.drift_detection import EDDM
    from skmultiflow.drift_detection.adwin import ADWIN
    import wandb

    import matplotlib.pyplot as plt

    from data_streams import classification_datasets, regression_datasets, load_format_data, cls_num, reg_num

    # Metrics
    gama_metrics = {
        "acc": 'accuracy',
        "b_acc": "balanced_accuracy",
        "f1": "f1",
        "roc_auc": "roc_auc",
        "rmse": "neg_root_mean_squared_error",
        "mse": "neg_mean_squared_error"
    }

    online_metrics = {
        "acc": metrics.Accuracy(),
        "b_acc": metrics.BalancedAccuracy(),
        "f1": metrics.F1(),
        "roc_auc": metrics.ROCAUC(),
        "rmse": metrics.RMSE(),
        "mse": metrics.MSE()
    }

    # Search algorithms
    search_algs = {
        "random": RandomSearch(),
        "evol": AsyncEA(),
        "s_halving": AsynchronousSuccessiveHalving()
    }

    # User parameters

    # prints python_script.py
    print(sys.argv[0])
    # prints dataset no
    print(f"Data stream is {sys.argv[1]}.")
    # prints initial batch size
    print(f"Initial batch size is {int(sys.argv[2])}.")
    # prints sliding window size
    print(f"Sliding window size is {int(sys.argv[3])}.")
    # prints gama performance metric
    print(f"Gama performance metric is {gama_metrics[str(sys.argv[4])]}.")
    # prints online performance metric
    print(f"Online performance metric is {online_metrics[str(sys.argv[5])]}.")
    # prints time budget for GAMA
    print(f"Time budget for GAMA is {int(sys.argv[6])}.")
    # prints search algorithm for GAMA
    print(f"Search algorithm for GAMA is {search_algs[str(sys.argv[7])]}.")
    print(f"Live plotting (wandb) is {eval(sys.argv[8])}.")

    data_loc = sys.argv[1]  # needs to be arff
    initial_batch = int(sys.argv[2])  # initial set of samples to train automl
    # update set of samples to train automl at drift points (must be smaller than or equal to initial batch size
    sliding_window = int(sys.argv[3])
    # gama metric to evaluate in pipeline search
    gama_metric = gama_metrics[str(sys.argv[4])]
    # river metric to evaluate online learning
    online_metric = online_metrics[str(sys.argv[5])]
    time_budget = int(sys.argv[6])  # time budget for gama run
    search_alg = search_algs[str(sys.argv[7])]
    live_plot = eval(sys.argv[8])

    # verify dataset is abailable
    assert data_loc in regression_datasets + \
        classification_datasets, f"Path to dataset '{data_loc}' not found"

    # set task for given dataset
    if data_loc in regression_datasets:
        task = "regression"
        ModelPipeline = GamaRegressor
        is_classification = False
        WANDB_PROJECT = "Results"
        assert str(sys.argv[5]) in [
            "rmse", "mse", "mae"], f"""Metric: '{str(sys.argv[5])}' not compatible with '{task}' """

    elif data_loc in classification_datasets:
        task = "classification"
        ModelPipeline = GamaClassifier
        is_classification = True
        WANDB_PROJECT = "OAML Classification"
        assert str(sys.argv[5]) in [
            "acc", "b_acc", "f1", "roc_auc"], f"""Metric: '{str(sys.argv[5])}' not compatible with '{task}' """

    # EDDM only compatible with classification
    drift_detector = EDDM() if is_classification else ADWIN()

    # Plot initialization
    if live_plot:
        table_data = []
        os.environ["WANDB_START_METHOD"] = "thread"
        wandb.init(
            name="Extended OAML - Ensemble (DWM)",
            project="OAML Classification",
            entity="lota",
            tags=["classification", "ensemble", "search"],
            config={
                "dataset": data_loc,
                "batch_size": initial_batch,
                "sliding_window": sliding_window,
                "tags": "DWM Ensemble",
                "gama_performance_metric": gama_metric,
                "online_performance_metric": online_metric,
                "time_budget_gama": time_budget,
                "search_algorithm": search_alg
            })

    # Data Loading
    X, y = load_format_data(data_loc, task, target_last=True)

    retraining_point = 50000 if is_classification else 300
    refractory_period = 1000 if is_classification else 100
    logging_point = int(len(X) * 0.01)

    # Algorithm selection and hyperparameter tuning
    Auto_pipeline = ModelPipeline(max_total_time=time_budget,
                                  scoring=gama_metric,
                                  search=search_alg,
                                  online_learning=True,
                                  post_processing=BestFitOnlinePostProcessing(),
                                  store='nothing',
                                  )

    Auto_pipeline.fit(X.iloc[0:initial_batch], y[0:initial_batch])
    print(
        f'Initial model is {Auto_pipeline.model} and hyperparameters are: {Auto_pipeline.model._get_params()}')
    print("Online model is updated with latest AutoML pipeline.")

    # Online learning
    if is_classification:
        Backup_ensemble = ensemble.AdaptiveVotingClassifier(
            [Auto_pipeline.model], dwm=0)  # [0=dwm, 1=addexp, 2=unweighteed]
    else:
        Backup_ensemble = ensemble.AdaptiveVotingRegressor(
            [Auto_pipeline.model], weighted=True)

    Online_model = Auto_pipeline.model
    count_drift = 0
    last_training_point = initial_batch
    online_stats = StreamStats(initial_batch)

    if live_plot:
        table_data.append([0, str(Online_model),
                           str(Online_model._get_params())])

    print(f'Test batch - 0 with RMSE: 0')
    for i in range(initial_batch + 1, len(X)):

        # handle bad inputs for target
        if not is_classification:
            y[i] = online_stats.validate_y(y[i])

        # Test then train - by one
        y_pred = Online_model.predict_one(X.iloc[i].to_dict())
        online_metric = online_metric.update(y[i], y_pred)
        Online_model = Online_model.learn_one(X.iloc[i].to_dict(), y[i])

        # Print performance every x interval
        if i % logging_point == 0:
            print(f'Test batch - {i} with {online_metric}')
            if live_plot:
                wandb.log(
                    {"current_point": i, "Prequential performance": online_metric.get()})

        # Check for drift
        if is_classification:
            drift_detector.add_element(int(y_pred != y[i]))
        else:
            drift_detector.add_element(y[i])

        if (drift_detector.detected_change()) or ((i - last_training_point) > retraining_point):
            if i - last_training_point < refractory_period:
                continue
            if drift_detector.detected_change():
                print(
                    f"Change detected at data point {i} and current performance is at {online_metric}")
                if live_plot:
                    wandb.log({"drift_point": i, "current_point": i,
                               "Prequential performance": online_metric.get()})

            elif (i - last_training_point) > retraining_point:
                print(
                    f"No drift but retraining point {i} and current performance is at {online_metric}")
                if live_plot:
                    wandb.log({"drift_point": i, "current_point": i,
                               "Prequential performance": online_metric.get()})

            last_training_point = i

            # Sliding window at the time of drift
            X_sliding = X.iloc[(i - sliding_window):i].reset_index(drop=True)
            y_sliding = y[(i - sliding_window):i].reset_index(drop=True)

            # re-optimize pipelines with sliding window
            Auto_pipeline = ModelPipeline(max_total_time=time_budget,
                                          scoring=gama_metric,
                                          search=search_alg,
                                          online_learning=True,
                                          post_processing=BestFitOnlinePostProcessing(),
                                          store='nothing',
                                          )
            Auto_pipeline.fit(X_sliding, y_sliding)

            # Ensemble performance comparison
            if is_classification:
                Perf_ensemble, Perf_automodel = metrics.Accuracy(), metrics.Accuracy()
            else:
                Perf_ensemble, Perf_automodel = metrics.RMSE(), metrics.RMSE()

            for xi, yi in stream.iter_pandas(X_sliding, y_sliding):
                y_pred_ensemble = Backup_ensemble.predict_one(xi)
                y_pred_online = Auto_pipeline.model.predict_one(xi)

                Perf_ensemble = Perf_ensemble.update(yi, y_pred_ensemble)
                Perf_automodel = Perf_automodel.update(yi, y_pred_online)

                Backup_ensemble = Backup_ensemble.learn_one(xi, yi)
                Auto_pipeline.model = Auto_pipeline.model.learn_one(xi, yi)

            print(f"Batch Performance on Ensemble: {Perf_ensemble}")
            print(f"Batch Performance on current model: {Perf_automodel}")

            # compare performance for given task
            select_ensemble = select_ensemble_from(
                Perf_automodel.get(), Perf_ensemble.get(), is_classification)

            if select_ensemble:
                Online_model = Backup_ensemble
                if live_plot:
                    wandb.log({"current_point": i, "model_update": 2})
            else:
                Online_model = Auto_pipeline.model
                if live_plot:
                    wandb.log({"current_point": i, "model_update": 1})

            # Ensemble update with new model, remove oldest model if ensemble is full
            Backup_ensemble.models.append(Auto_pipeline.model)

            # Replace least performing model in ensemble
            if len(Backup_ensemble.models) > 10:
                # Backup_ensemble.models.pop(0)
                Backup_ensemble.pop_worst()

            print(
                f'Current model is {Online_model} and hyperparameters are: {Online_model._get_params()}')
            if live_plot:
                table_data.append([i, str(Online_model),
                                   str(Online_model._get_params())])

    if live_plot:
        model_table = wandb.Table(data=table_data, columns=[
                                  "i", "model_name", "hyperparameters"])
        wandb.log({"Selected pipelines": model_table})

        # elif (Online_model == Backup_ensemble) and (Backup_ensemble.get_loss() > Backup_ensemble.threshold):
