import numpy as np

from model import (
    cooperative_squared_error,
    expected_individual_squared_error,
    make_default_model,
    stable_softmax,
)


X = np.array([[1.0]])
Y = np.array([[1.0]])


def test_stable_softmax_sums_to_one_and_handles_large_values():
    values = stable_softmax(np.array([[10_000.0, 9_999.0]]))
    np.testing.assert_allclose(values.sum(axis=1), 1.0)
    assert np.all(np.isfinite(values))


def test_temperature_controls_sharpness():
    logits = np.array([[2.0, 0.0]])
    cold = stable_softmax(logits, 0.5)[0, 0]
    warm = stable_softmax(logits, 2.0)[0, 0]
    assert cold > warm > 0.5


def test_soft_and_hard_predictions():
    model = make_default_model()
    assert model.predict_soft(X)[0, 0] == pytest.approx(0.76)
    prediction, index = model.predict_hard(X, return_indices=True)
    assert index[0] == 0
    assert prediction[0, 0] == pytest.approx(0.9)


def test_stochastic_selection_is_categorical_and_does_not_update():
    model = make_default_model(seed=7)
    before = model.parameter_snapshot()
    repeated = np.ones((20_000, 1))
    _, indices = model.predict_stochastic(repeated, return_indices=True)
    frequency = np.mean(indices == 0)
    assert frequency == pytest.approx(0.8, abs=0.015)
    after = model.parameter_snapshot()
    for old, new in zip(before["expert_weights"], after["expert_weights"]):
        np.testing.assert_array_equal(old, new)
    np.testing.assert_array_equal(before["gate_weight"], after["gate_weight"])


def test_paper_equations_and_cancellation_case():
    outputs = np.array([[[0.9], [0.2]]])
    probabilities = np.array([[0.8, 0.2]])
    assert cooperative_squared_error(Y, outputs, probabilities)[0] == pytest.approx(0.0576)
    assert expected_individual_squared_error(Y, outputs, probabilities)[0] == pytest.approx(0.136)

    cancellation = np.array([[[0.9], [1.4]]])
    assert cooperative_squared_error(Y, cancellation, probabilities)[0] == pytest.approx(0.0)
    assert expected_individual_squared_error(Y, cancellation, probabilities)[0] == pytest.approx(0.04)


def test_equation_1_3_at_paper_scale():
    model = make_default_model(sigma=1.0)
    assert model.compute_competitive_loss(X, Y) == pytest.approx(0.06055735, abs=1e-8)


def test_default_extended_loss_and_responsibilities():
    model = make_default_model()
    assert model.compute_competitive_loss(X, Y) == pytest.approx(0.17463152, abs=1e-8)
    h = model.compute_responsibilities(X, Y)[0]
    np.testing.assert_allclose(h, [0.93378223, 0.06621777], atol=1e-8)
    assert np.all(h >= 0)
    assert h.sum() == pytest.approx(1.0)


def test_better_fit_can_overcome_lower_gate_prior():
    model = make_default_model()
    # Expert 2 begins with only 20% gate probability, but is now an exact fit.
    model.experts[0].weight[...] = 0.0
    model.experts[1].weight[...] = 1.0
    h = model.compute_responsibilities(X, Y)[0]
    assert h[1] > h[0]


def test_inference_needs_no_target_but_training_does():
    model = make_default_model()
    assert model.predict_soft(X).shape == (1, 1)
    with pytest.raises(TypeError):
        model.train_step(X)  # type: ignore[call-arg]


# Kept at the bottom so the test file documents all numerical imports clearly.
import pytest
