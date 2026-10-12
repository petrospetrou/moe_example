import numpy as np
import pytest

from model import make_default_model


X = np.array([[1.0]])
Y = np.array([[1.0]])


def finite_difference(model, parameter, index, epsilon=1e-6):
    old = parameter[index]
    parameter[index] = old + epsilon
    plus = model.compute_competitive_loss(X, Y)
    parameter[index] = old - epsilon
    minus = model.compute_competitive_loss(X, Y)
    parameter[index] = old
    return (plus - minus) / (2 * epsilon)


def test_expert_gradient_matches_finite_difference():
    model = make_default_model()
    analytical = model.compute_gradients(X, Y)["expert_weight_grads"][0][0, 0]
    numerical = finite_difference(model, model.experts[0].weight, (0, 0))
    assert analytical == pytest.approx(numerical, rel=1e-6, abs=1e-7)


def test_gate_raw_logit_gradient_matches_finite_difference():
    model = make_default_model()
    # The bias is a direct raw-logit parameter, so its derivative checks dL/dz.
    analytical = model.compute_gradients(X, Y)["gate_bias_grad"][0]
    numerical = finite_difference(model, model.gate.bias, (0,))
    assert analytical == pytest.approx(numerical, rel=1e-6, abs=1e-7)


def test_one_step_matches_acceptance_values_and_updates_all_parameters():
    model = make_default_model()
    original = model.parameter_snapshot()
    result = model.train_step(X, Y)

    grads = result["gradients"]["expert_output_grads"][0, :, 0]
    np.testing.assert_allclose(grads, [-0.37351289, -0.21189687], atol=1e-8)
    outputs = result["after"]["expert_outputs"][0, :, 0]
    np.testing.assert_allclose(outputs, [0.97470258, 0.24237937], atol=1e-8)
    probabilities = result["after"]["gate_probs"][0]
    np.testing.assert_allclose(probabilities, [0.80842480, 0.19157520], atol=1e-8)
    assert result["soft_prediction_after"][0, 0] == pytest.approx(0.83440762, abs=1e-8)
    assert result["after_loss"] == pytest.approx(0.14136396, abs=1e-8)

    for i, expert in enumerate(model.experts):
        assert not np.array_equal(expert.weight, original["expert_weights"][i])
        assert not np.array_equal(expert.bias, original["expert_biases"][i])
    assert not np.array_equal(model.gate.weight, original["gate_weight"])
    assert not np.array_equal(model.gate.bias, original["gate_bias"])


def test_reset_restores_all_parameters_and_rng():
    model = make_default_model(seed=123)
    original = model.parameter_snapshot()
    model.train_step(X, Y)
    model.reset()
    restored = model.parameter_snapshot()
    for key in ("expert_weights", "expert_biases"):
        for expected, actual in zip(original[key], restored[key]):
            np.testing.assert_array_equal(expected, actual)
    np.testing.assert_array_equal(original["gate_weight"], restored["gate_weight"])
    np.testing.assert_array_equal(original["gate_bias"], restored["gate_bias"])


def test_batch_shapes_and_vector_output_broadcasting():
    # The tutorial is scalar, but this catches accidental scalar-only math.
    model = make_default_model()
    batch = np.array([[1.0], [0.5], [-1.0]])
    targets = np.array([[1.0], [0.0], [0.0]])
    assert model.forward(batch)["expert_outputs"].shape == (3, 2, 1)
    assert model.compute_responsibilities(batch, targets).shape == (3, 2)
    assert np.isfinite(model.compute_competitive_loss(batch, targets))
