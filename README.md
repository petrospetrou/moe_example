# The Cat Committee: Adaptive Mixtures of Local Experts

An interactive, beginner-first Streamlit lesson about the model introduced in:

> Jacobs, R. A., Jordan, M. I., Nowlan, S. J., & Hinton, G. E. (1991). “Adaptive Mixtures of Local Experts.” *Neural Computation*, 3(1), 79–87.

The app follows one continuous cat-detection story across 22 ordered steps. It begins with the difference between inference and training, then builds soft, hard, and stochastic prediction, the paper’s three objectives, posterior responsibilities, real NumPy gradients, one reproducible learning update, and a small specialization experiment.

## Install and launch

From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit, normally `http://localhost:8501`.

Use **Previous**, **Reset**, and **Next** at the bottom of the card. Only one lesson step appears at a time. Reset restarts the walkthrough at Step 1 and restores every deterministic default.

## What is actually implemented

`model.py` contains a real NumPy implementation of:

- `LinearExpert`, with `o = xW + b`;
- `GatingNetwork`, with stable temperature-scaled softmax;
- `MixtureOfExperts`, with separate soft, hard, and stochastic inference methods;
- stable mixture likelihood and posterior responsibilities;
- analytical expert and gate gradients;
- simultaneous SGD updates to every expert weight and bias and to the gate weight matrix and bias;
- deterministic reset and stochastic sampling.

The primary example always uses `x = 1`, target `y = 1`, expert scores `[0.90, 0.20]`, gate probabilities `[0.80, 0.20]`, `sigma = 0.5`, `tau = 1`, and learning rate `0.1`. Both expert weights and biases are updated, so at `x = 1` both parameter changes contribute to each new output.

## Paper equations and teaching extensions

The lesson explicitly introduces the paper’s progression:

- Equation 1.1: cooperative squared error after combining expert outputs;
- Equation 1.2: expected individual expert squared error;
- Equation 1.3: Gaussian-mixture negative log-likelihood;
- Equation 1.4: the expected-error expert-output derivative;
- Equation 1.5: the mixture-likelihood expert-output derivative using posterior responsibility.

It also explains Figure 1’s stochastic-selector interpretation and Section 2’s distinction between soft and hard competitive learning.

The following are clearly labelled teaching or implementation extensions, not claims about the original experiment:

- cat detection instead of vowel recognition;
- two scalar linear experts and a conceptual scalar input;
- the numerical scores, initial parameters, and one-step update;
- adjustable Gaussian scale `sigma`;
- adjustable softmax temperature `tau`;
- adjustable learning rate;
- a small deterministic synthetic cat/no-cat dataset;
- soft weighted prediction as the tutorial’s primary output rule.

The expert outputs are unconstrained linear scores. Values between zero and one are sometimes described intuitively as percentages, but a score above one is always identified as a raw score rather than a probability.

## Run the tests

```bash
python -m pytest -q
```

The tests cover stable softmax, temperature, all inference strategies, stochastic non-mutation, Equations 1.1–1.3, the cancellation counterexample, responsibilities, finite-difference checks of both gradient families, all parameter updates, exact numerical acceptance values, reset behavior, target requirements, every lesson screen, and navigation interactions.

## Scientific limitations

- This is an educational scalar model, not a cat image classifier. It has no image encoder and does not inspect real pictures.
- It is not the authors’ original code and does not reproduce their vowel-recognition experiment or report their experimental metrics.
- The synthetic specialization experiment begins with deliberately asymmetric experts and gate parameters. It demonstrates a possible competitive-learning trajectory; it does not promise that specialization emerges from every initialization or dataset.
- Gaussian normalization constants are omitted exactly where they do not affect optimization under fixed variance, matching the displayed unit-scale form used for instruction.
- Modern sparse top-k transformer MoEs are outside this lesson’s scope.

## Project map

- `app.py` — lesson state, all 22 screens, and interactions
- `model.py` — numerical model and training logic
- `components.py` — small reusable visual components
- `styles.py` — scoped dark theme and naturally growing lesson card
- `lesson_content.py` — ordered lesson metadata and citation
- `tests/` — mathematical, training, rendering, and navigation tests
