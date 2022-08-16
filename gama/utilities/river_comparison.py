

def replace_current_model(
    current_performance: float,
    new_performance: float,
    is_classification: bool
) -> bool:
    """ Determine whether new models performance is better than the online current model on given task.

    Parameters
    ----------
    current_performance: float
        Perfornance of current model to be compared.
    new_performance: float
        Performance of new model to be compared.
    is_classification: bool
        If True, task is a classification task
        If False, task is a regression task.


    Returns
    -------
    A boolean, 
    if True new_performance is better than current_performance for given task.
    if False current_performance is better than new_performance for given task
    """
    if is_classification:
        print(current_performance, new_performance)
        if new_performance >= current_performance:  # higher is better for accuracy metrics
            replacement = True
            print("Online model is updated with latest AutoML pipeline.")
        else:
            print("Online model is kept at current AutoML pipeline.")
            replacement = False

    # Minimize metric for regression, maximise metric for classification
    else:
        if new_performance <= current_performance:  # Lower is better for loss metrics
            print("Online model is updated with latest AutoML pipeline.")
            replacement = True
        else:
            print("Online model is kept at current AutoML pipeline.")
            replacement = False

    return replacement


def select_ensemble_from(
    online_model_performance: float,
    ensemble_performance: float,
    is_classification: bool
) -> bool:
    """ Determine whether an ensembles performance is better than single model on given task.

    Parameters
    ----------
    online_model_performance: float
        Perfornance of single model to be compared.
    ensemble_performance: float
        Performance of ensemble model to be compared.
    is_classification: bool
        If True, task is a classification task
        If False, task is a regression task.

    Returns
    -------
    A boolean, 
    if True ensemble_performance is better than online_model_performance for given task.
    if False online_model_performance is better than ensemble_performance for given task
    """
    if is_classification:
        if ensemble_performance > online_model_performance:  # higher is better for accuracy metrics
            select_ensemble = True
            print("Online model is updated with Backup Ensemble.")
        else:
            print("Online model is updated with latest AutoML pipeline.")
            select_ensemble = False

    # Minimize metric for regression, maximise metric for classification
    else:
        if ensemble_performance <= online_model_performance:  # Lower is better for loss metrics
            print("Online model is updated with Backup Ensemble.")
            select_ensemble = True
        else:
            print("Online model is updated with latest AutoML pipeline.")
            select_ensemble = False

    return select_ensemble
