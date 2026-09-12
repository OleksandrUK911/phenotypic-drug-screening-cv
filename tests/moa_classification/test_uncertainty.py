import torch

from src.moa_classification.model import MOAClassifier
from src.moa_classification.uncertainty import ensemble_predict, mc_dropout_predict


def test_mc_dropout_predict_output_shapes_and_valid_probs():
    torch.manual_seed(0)
    model = MOAClassifier(input_dim=16, hidden_dim=32, n_classes=4, dropout=0.5)
    x = torch.randn(5, 16)

    result = mc_dropout_predict(model, x, n_samples=10)

    assert result.mean_probs.shape == (5, 4)
    assert torch.allclose(result.mean_probs.sum(dim=-1), torch.ones(5), atol=1e-4)
    assert result.predictive_entropy.shape == (5,)
    assert (result.predictive_entropy >= 0).all()
    assert (result.epistemic_uncertainty >= 0).all()


def test_mc_dropout_epistemic_uncertainty_zero_when_dropout_disabled():
    torch.manual_seed(0)
    model = MOAClassifier(input_dim=8, hidden_dim=16, n_classes=3, dropout=0.0)
    x = torch.randn(3, 8)

    result = mc_dropout_predict(model, x, n_samples=15)
    assert torch.allclose(result.epistemic_uncertainty, torch.zeros(3), atol=1e-6)


def test_ensemble_predict_output_shapes():
    torch.manual_seed(0)
    models = [MOAClassifier(input_dim=10, hidden_dim=16, n_classes=5) for _ in range(4)]
    x = torch.randn(6, 10)

    result = ensemble_predict(models, x)
    assert result.mean_probs.shape == (6, 5)
    assert torch.allclose(result.mean_probs.sum(dim=-1), torch.ones(6), atol=1e-4)
