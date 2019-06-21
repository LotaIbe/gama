from functools import partial
import logging
import math

from gama.utilities.generic.async_executor import AsyncExecutor, wait_first_complete
from gama.genetic_programming.compilers.scikitlearn import evaluate_individual
from gama.genetic_programming.algorithms.metrics import Metric
from gama.utilities.preprocessing import define_preprocessing_steps
"""
TODO:
 - instead of list, use a min-heap by rung.
 - promoted pipelines as set and set-intersection to determine promotability?
"""

log = logging.getLogger(__name__)


def asha(operations, start_candidates=None, timeout=300,  # General Search Hyperparameters
         reduction_factor=3, minimum_resource=100, maximum_resource=1700, minimum_early_stopping_rate=1):  # Algorithm Specific
    # Note that here we index the rungs by all possible rungs (0..ceil(log_eta(R/r))), and ignore the first
    # minimum_early_stopping_rate rungs. This contrasts the paper where rung 0 refers to the first used one.
    max_rung = math.ceil(math.log(maximum_resource/minimum_resource, reduction_factor))
    rungs = range(minimum_early_stopping_rate, max_rung + 1)
    resource_for_rung = {rung: min(minimum_resource * (reduction_factor ** rung), maximum_resource) for rung in rungs}

    # Should we just use lists of lists/heaps instead?
    individuals_by_rung = {rung: [] for rung in reversed(rungs)}  # Highest rungs first is how we typically access them
    promoted_individuals = {rung: [] for rung in reversed(rungs)}

    def get_job():
        for rung, individuals in list(individuals_by_rung.items())[1:]:
            # This is not in the paper code but is derived from fig 2b
            n_to_promote = math.floor(len(individuals) / reduction_factor)
            if n_to_promote - len(promoted_individuals[rung]) > 0:
                # Problem: equal loss falls back on comparison of individual
                candidates = list(sorted(individuals, key=lambda t: t[0], reverse=True))[:n_to_promote]
                promotable = [candidate for candidate in candidates if candidate not in promoted_individuals[rung]]
                if len(promotable) > 0:
                    promoted_individuals[rung].append(promotable[0])
                    return promotable[0][1], rung + 1

        if start_candidates is not None and len(start_candidates) > 0:
            return start_candidates.pop(), minimum_early_stopping_rate
        else:
            return operations.individual(), minimum_early_stopping_rate

    futures = set()
    with AsyncExecutor() as async_:
        for _ in range(8):
            individual, rung = get_job()
            futures.add(async_.submit(operations.evaluate, individual, rung, subsample=resource_for_rung[rung]))

        for _ in range(100):
            done, futures = wait_first_complete(futures)
            for loss, individual, rung in [future.result() for future in done]:
                individuals_by_rung[rung].append((loss, individual))
                print("[{}] {}: {}".format(rung, individual.pipeline_str(), loss))
            individual, rung = get_job()
            print("Putting individual in rung {}".format(rung))
            n = minimum_resource * (reduction_factor ** (minimum_early_stopping_rate + rung))
            futures.add(async_.submit(operations.evaluate, individual, rung, subsample=n))

    highest_rung_reached = max(rung for rung, individuals in individuals_by_rung.items() if individuals != [])
    for rung, individuals in individuals_by_rung.items():
        print('[{}] {}'.format(rung, len(individuals)))
    if highest_rung_reached != max(rungs):
        raise RuntimeWarning("Highest rung not reached.")

    return list(map(lambda p: p[1], individuals_by_rung[highest_rung_reached]))


def evaluate_on_rung(individual, rung, *args, **kwargs):
    individual = evaluate_individual(individual, *args, **kwargs)
    return individual.fitness.values[0], individual, rung


if __name__ == '__main__':
    from gama import GamaClassifier
    from sklearn.datasets import load_digits
    g = GamaClassifier()
    X, y = load_digits(return_X_y=True)

    steps = define_preprocessing_steps(X, max_extra_features_created=None, max_categories_for_one_hot=10)
    g._operator_set._safe_compile = partial(g._operator_set._compile, preprocessing_steps=steps)
    g._operator_set.evaluate = partial(evaluate_on_rung, evaluate_pipeline_length=False, X=X, y_train=y, y_score=y, timeout=300, metrics=[Metric.from_string('accuracy')])
    start_pop = [g._operator_set.individual() for _ in range(10)]
    pipeline = asha(g._operator_set, start_candidates=start_pop)
    print(pipeline)
