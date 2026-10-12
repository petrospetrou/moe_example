"""Small, transparent NumPy implementation of an adaptive mixture of experts.

The module intentionally keeps inference and training separate.  It supports
batched inputs, multiple linear experts, and vector-valued expert outputs,
although the tutorial uses one sample, one input, and one output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


Array = np.ndarray


def _positive(value: float, name: str) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return value


def stable_softmax(values: Array, temperature: float = 1.0, axis: int = -1) -> Array:
    """Numerically stable temperature-scaled softmax."""

    tau = _positive(temperature, "temperature")
    scaled = np.asarray(values, dtype=float) / tau
    shifted = scaled - np.max(scaled, axis=axis, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=axis, keepdims=True)


def logsumexp(values: Array, axis: int = -1, keepdims: bool = False) -> Array:
    """A dependency-free, stable log-sum-exp."""

    values = np.asarray(values, dtype=float)
    maximum = np.max(values, axis=axis, keepdims=True)
    result = maximum + np.log(np.sum(np.exp(values - maximum), axis=axis, keepdims=True))
    return result if keepdims else np.squeeze(result, axis=axis)


def cooperative_squared_error(target: Array, expert_outputs: Array, gate_probs: Array) -> Array:
    """Paper equation (1.1): squared error after combining outputs."""

    target = np.asarray(target, dtype=float)
    expert_outputs = np.asarray(expert_outputs, dtype=float)
    gate_probs = np.asarray(gate_probs, dtype=float)
    combined = np.sum(gate_probs[..., None] * expert_outputs, axis=-2)
    return np.sum((target - combined) ** 2, axis=-1)


def expected_individual_squared_error(
    target: Array, expert_outputs: Array, gate_probs: Array
) -> Array:
    """Paper equation (1.2): gate-weighted individual squared errors."""

    target = np.asarray(target, dtype=float)
    expert_outputs = np.asarray(expert_outputs, dtype=float)
    gate_probs = np.asarray(gate_probs, dtype=float)
    distances = np.sum((target[..., None, :] - expert_outputs) ** 2, axis=-1)
    return np.sum(gate_probs * distances, axis=-1)


@dataclass
class LinearExpert:
    """A single expert with ``output = input @ weight + bias``."""

    weight: Array
    bias: Array

    def __post_init__(self) -> None:
        self.weight = np.asarray(self.weight, dtype=float).copy()
        self.bias = np.asarray(self.bias, dtype=float).copy()
        if self.weight.ndim != 2 or self.bias.ndim != 1:
            raise ValueError("Expert weight must be 2-D and bias must be 1-D")
        if self.weight.shape[1] != self.bias.shape[0]:
            raise ValueError("Expert weight and bias output dimensions must agree")

    def forward(self, inputs: Array) -> Array:
        inputs = np.atleast_2d(np.asarray(inputs, dtype=float))
        return inputs @ self.weight + self.bias


@dataclass
class GatingNetwork:
    """Linear logits followed by a temperature-scaled softmax."""

    weight: Array
    bias: Array

    def __post_init__(self) -> None:
        self.weight = np.asarray(self.weight, dtype=float).copy()
        self.bias = np.asarray(self.bias, dtype=float).copy()
        if self.weight.ndim != 2 or self.bias.ndim != 1:
            raise ValueError("Gate weight must be 2-D and bias must be 1-D")
        if self.weight.shape[1] != self.bias.shape[0]:
            raise ValueError("Gate weight and bias expert dimensions must agree")

    def raw_logits(self, inputs: Array) -> Array:
        inputs = np.atleast_2d(np.asarray(inputs, dtype=float))
        return inputs @ self.weight + self.bias

    def forward(self, inputs: Array, tau: float = 1.0) -> Array:
        return stable_softmax(self.raw_logits(inputs), temperature=tau, axis=-1)


class MixtureOfExperts:
    """A minimal adaptive mixture with explicit prediction and training APIs."""

    def __init__(
        self,
        experts: list[LinearExpert],
        gate: GatingNetwork,
        *,
        sigma: float = 0.5,
        tau: float = 1.0,
        learning_rate: float = 0.1,
        seed: int = 1991,
    ) -> None:
        if not experts:
            raise ValueError("At least one expert is required")
        if gate.bias.size != len(experts):
            raise ValueError("The gate must produce one probability per expert")
        output_dims = {expert.bias.size for expert in experts}
        input_dims = {expert.weight.shape[0] for expert in experts} | {gate.weight.shape[0]}
        if len(output_dims) != 1 or len(input_dims) != 1:
            raise ValueError("All network dimensions must agree")

        self.experts = experts
        self.gate = gate
        self.sigma = _positive(sigma, "sigma")
        self.tau = _positive(tau, "tau")
        self.learning_rate = _positive(learning_rate, "learning_rate")
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self._initial_state = self.parameter_snapshot()

    @property
    def num_experts(self) -> int:
        return len(self.experts)

    def parameter_snapshot(self) -> dict[str, Any]:
        return {
            "expert_weights": [expert.weight.copy() for expert in self.experts],
            "expert_biases": [expert.bias.copy() for expert in self.experts],
            "gate_weight": self.gate.weight.copy(),
            "gate_bias": self.gate.bias.copy(),
        }

    def reset(self) -> None:
        for expert, weight, bias in zip(
            self.experts,
            self._initial_state["expert_weights"],
            self._initial_state["expert_biases"],
        ):
            expert.weight[...] = weight
            expert.bias[...] = bias
        self.gate.weight[...] = self._initial_state["gate_weight"]
        self.gate.bias[...] = self._initial_state["gate_bias"]
        self.rng = np.random.default_rng(self.seed)

    @staticmethod
    def _inputs(inputs: Array) -> Array:
        values = np.asarray(inputs, dtype=float)
        return values.reshape(1, -1) if values.ndim == 1 else values

    @staticmethod
    def _targets(targets: Array) -> Array:
        values = np.asarray(targets, dtype=float)
        return values.reshape(1, -1) if values.ndim == 1 else values

    def forward(self, inputs: Array) -> dict[str, Array]:
        """Inference-only forward pass; no target is needed and nothing changes."""

        inputs = self._inputs(inputs)
        outputs = np.stack([expert.forward(inputs) for expert in self.experts], axis=1)
        logits = self.gate.raw_logits(inputs)
        probabilities = stable_softmax(logits, self.tau)
        return {"expert_outputs": outputs, "logits": logits, "gate_probs": probabilities}

    def predict_soft(self, inputs: Array) -> Array:
        forward = self.forward(inputs)
        return np.sum(forward["gate_probs"][..., None] * forward["expert_outputs"], axis=1)

    def predict_hard(self, inputs: Array, *, return_indices: bool = False) -> Array | tuple[Array, Array]:
        forward = self.forward(inputs)
        indices = np.argmax(forward["gate_probs"], axis=1)
        predictions = forward["expert_outputs"][np.arange(indices.size), indices]
        return (predictions, indices) if return_indices else predictions

    def predict_stochastic(
        self,
        inputs: Array,
        *,
        rng: np.random.Generator | None = None,
        return_indices: bool = False,
    ) -> Array | tuple[Array, Array]:
        forward = self.forward(inputs)
        generator = self.rng if rng is None else rng
        indices = np.array([generator.choice(self.num_experts, p=row) for row in forward["gate_probs"]])
        predictions = forward["expert_outputs"][np.arange(indices.size), indices]
        return (predictions, indices) if return_indices else predictions

    def compute_squared_expert_errors(
        self, inputs: Array, targets: Array, *, forward: dict[str, Array] | None = None
    ) -> Array:
        targets = self._targets(targets)
        forward = self.forward(inputs) if forward is None else forward
        return np.sum((targets[:, None, :] - forward["expert_outputs"]) ** 2, axis=-1)

    def compute_responsibilities(
        self, inputs: Array, targets: Array, *, forward: dict[str, Array] | None = None
    ) -> Array:
        sigma = _positive(self.sigma, "sigma")
        forward = self.forward(inputs) if forward is None else forward
        distances = self.compute_squared_expert_errors(inputs, targets, forward=forward)
        log_terms = np.log(forward["gate_probs"]) - distances / (2.0 * sigma**2)
        return np.exp(log_terms - logsumexp(log_terms, axis=1, keepdims=True))

    def compute_competitive_loss(
        self, inputs: Array, targets: Array, *, forward: dict[str, Array] | None = None
    ) -> float:
        sigma = _positive(self.sigma, "sigma")
        forward = self.forward(inputs) if forward is None else forward
        distances = self.compute_squared_expert_errors(inputs, targets, forward=forward)
        log_terms = np.log(forward["gate_probs"]) - distances / (2.0 * sigma**2)
        return float(np.mean(-logsumexp(log_terms, axis=1)))

    def compute_gradients(self, inputs: Array, targets: Array) -> dict[str, Any]:
        """Gradients of the mean competitive loss for all parameters."""

        sigma = _positive(self.sigma, "sigma")
        tau = _positive(self.tau, "tau")
        inputs = self._inputs(inputs)
        targets = self._targets(targets)
        if inputs.shape[0] != targets.shape[0]:
            raise ValueError("inputs and targets must have the same batch size")
        forward = self.forward(inputs)
        responsibilities = self.compute_responsibilities(inputs, targets, forward=forward)
        batch_size = inputs.shape[0]

        grad_outputs = (
            responsibilities[..., None]
            * (forward["expert_outputs"] - targets[:, None, :])
            / sigma**2
            / batch_size
        )
        expert_weight_grads = [inputs.T @ grad_outputs[:, i, :] for i in range(self.num_experts)]
        expert_bias_grads = [np.sum(grad_outputs[:, i, :], axis=0) for i in range(self.num_experts)]

        grad_logits = (forward["gate_probs"] - responsibilities) / tau / batch_size
        gate_weight_grad = inputs.T @ grad_logits
        gate_bias_grad = np.sum(grad_logits, axis=0)

        return {
            "forward": forward,
            "responsibilities": responsibilities,
            "expert_output_grads": grad_outputs * batch_size,
            "expert_weight_grads": expert_weight_grads,
            "expert_bias_grads": expert_bias_grads,
            "gate_logit_grads": grad_logits * batch_size,
            "gate_weight_grad": gate_weight_grad,
            "gate_bias_grad": gate_bias_grad,
        }

    def train_step(self, inputs: Array, targets: Array) -> dict[str, Any]:
        """Perform one simultaneous SGD update and report before/after values."""

        eta = _positive(self.learning_rate, "learning_rate")
        before = self.forward(inputs)
        before_loss = self.compute_competitive_loss(inputs, targets, forward=before)
        gradients = self.compute_gradients(inputs, targets)
        for index, expert in enumerate(self.experts):
            expert.weight -= eta * gradients["expert_weight_grads"][index]
            expert.bias -= eta * gradients["expert_bias_grads"][index]
        self.gate.weight -= eta * gradients["gate_weight_grad"]
        self.gate.bias -= eta * gradients["gate_bias_grad"]
        after = self.forward(inputs)
        after_loss = self.compute_competitive_loss(inputs, targets, forward=after)
        return {
            "before": before,
            "after": after,
            "before_loss": before_loss,
            "after_loss": after_loss,
            "gradients": gradients,
            "soft_prediction_before": np.sum(
                before["gate_probs"][..., None] * before["expert_outputs"], axis=1
            ),
            "soft_prediction_after": np.sum(
                after["gate_probs"][..., None] * after["expert_outputs"], axis=1
            ),
        }


def make_default_model(
    *, sigma: float = 0.5, tau: float = 1.0, learning_rate: float = 0.1, seed: int = 1991
) -> MixtureOfExperts:
    """The reproducible one-sample model used throughout the lesson."""

    experts = [
        LinearExpert(np.array([[0.90]]), np.array([0.0])),
        LinearExpert(np.array([[0.20]]), np.array([0.0])),
    ]
    gate = GatingNetwork(np.array([[np.log(4.0), 0.0]]), np.array([0.0, 0.0]))
    return MixtureOfExperts(
        experts,
        gate,
        sigma=sigma,
        tau=tau,
        learning_rate=learning_rate,
        seed=seed,
    )


def make_specialization_model(seed: int = 1991) -> MixtureOfExperts:
    """Deterministic asymmetric model for the small multi-example experiment."""

    experts = [
        # An intentionally asymmetric start: one expert begins cat-leaning and
        # the other no-cat-leaning. Training must still perform the real work.
        LinearExpert(np.array([[0.0]]), np.array([0.85])),
        LinearExpert(np.array([[0.0]]), np.array([0.15])),
    ]
    gate = GatingNetwork(np.array([[0.50, -0.50]]), np.array([0.0, 0.0]))
    return MixtureOfExperts(experts, gate, sigma=0.5, tau=1.0, learning_rate=0.04, seed=seed)
