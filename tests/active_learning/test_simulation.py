from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from src.active_learning.simulation import simulate_active_learning


def _synthetic_split():
    x, y = make_classification(
        n_samples=300,
        n_features=10,
        n_informative=6,
        n_classes=3,
        n_clusters_per_class=1,
        class_sep=1.0,
        random_state=0,
    )
    return train_test_split(x, y, test_size=0.3, random_state=0)


def test_simulate_active_learning_returns_one_accuracy_per_round():
    x_train, x_test, y_train, y_test = _synthetic_split()

    accuracies = simulate_active_learning(
        x_train, y_train, x_test, y_test, strategy="random", n_initial=10, batch_size=10, n_rounds=5, seed=0
    )

    assert len(accuracies) == 6
    assert all(0.0 <= a <= 1.0 for a in accuracies)


def test_uncertainty_sampling_beats_random_on_fixed_synthetic_benchmark():
    """Deterministic comparison (fixed seeds, no test-time randomness): uncertainty-based
    compound prioritization should reach higher final accuracy than random screening order
    for the same number of "screened" compounds, on this benchmark.
    """
    x_train, x_test, y_train, y_test = _synthetic_split()

    random_accs = simulate_active_learning(
        x_train, y_train, x_test, y_test, strategy="random", n_initial=10, batch_size=10, n_rounds=5, seed=0
    )
    uncertainty_accs = simulate_active_learning(
        x_train, y_train, x_test, y_test, strategy="uncertainty", n_initial=10, batch_size=10, n_rounds=5, seed=0
    )

    assert uncertainty_accs[-1] > random_accs[-1]


def test_simulate_active_learning_rejects_unknown_strategy():
    x_train, x_test, y_train, y_test = _synthetic_split()
    try:
        simulate_active_learning(x_train, y_train, x_test, y_test, strategy="bogus")
    except ValueError:
        return
    raise AssertionError("Expected ValueError for unknown strategy")


def test_simulate_active_learning_stops_gracefully_when_pool_exhausted():
    x_train, x_test, y_train, y_test = _synthetic_split()
    accuracies = simulate_active_learning(
        x_train,
        y_train,
        x_test,
        y_test,
        strategy="random",
        n_initial=len(x_train) - 1,
        batch_size=5,
        n_rounds=10,
        seed=0,
    )
    assert len(accuracies) >= 1
