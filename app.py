"""Interactive, single-path lesson for Adaptive Mixtures of Local Experts."""

from __future__ import annotations

import math

import numpy as np
import streamlit as st

from components import (
    cat_stage,
    equation,
    history_chart,
    html_card,
    meters,
    source_note,
    takeaway,
    warning,
)
from lesson_content import LESSON_STEPS, PAPER_CITATION
from model import make_default_model, make_specialization_model
from styles import APP_CSS


X = np.array([[1.0]])
Y = np.array([[1.0]])
DEFAULT_OUTPUTS = np.array([0.9, 0.2])
DEFAULT_GATE = np.array([0.8, 0.2])
TOTAL_STEPS = len(LESSON_STEPS)


st.set_page_config(
    page_title="The Cat Committee · Adaptive Mixtures of Local Experts",
    page_icon="🐈",
    layout="centered",
    initial_sidebar_state="collapsed",
)
st.markdown(APP_CSS, unsafe_allow_html=True)


def initialize_state() -> None:
    defaults = {
        "lesson_step": 0,
        "stochastic_pick": None,
        "target_revealed": False,
        "simulation_result": None,
        "simulation_rng": np.random.default_rng(1991),
        "routing_rng": np.random.default_rng(1991),
        "selected_eta": 0.10,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def previous_step() -> None:
    st.session_state.lesson_step = max(0, st.session_state.lesson_step - 1)


def next_step() -> None:
    st.session_state.lesson_step = min(TOTAL_STEPS - 1, st.session_state.lesson_step + 1)


def reset_lesson() -> None:
    """Restart the walkthrough and restore every deterministic default."""

    st.session_state.clear()
    st.session_state.lesson_step = 0


def set_counterexample() -> None:
    st.session_state.counterexample_o2 = 1.40


def remember_learning_rate() -> None:
    st.session_state.selected_eta = float(st.session_state.main_eta_widget)


def default_training_preview(eta: float | None = None):
    model = make_default_model(learning_rate=st.session_state.get("selected_eta", 0.10) if eta is None else eta)
    result = model.train_step(X, Y)
    return model, result


def render_step_1() -> None:
    st.write("Let’s make this easy. A model only does one of two jobs at a time.")
    html_card(
        "INFERENCE · THE MODEL IS ANSWERING",
        "It uses what it already knows.",
        "We give it a question. It produces an answer. It does <b>not</b> learn or change its weights.",
        color="blue",
        css_class="phase-card",
    )
    html_card(
        "TRAINING · THE MODEL IS LEARNING",
        "It gets the question and the correct answer.",
        "It sees how well it did. Then it changes its weights to try to do better next time.",
        color="green",
        css_class="phase-card",
    )
    st.markdown("**Training = learning.**  \n**Inference = answering.**")
    answer = st.radio(
        "What happens when the correct answer is NOT available?",
        ["Choose an answer…", "Inference", "Training"],
        key="quiz_phase",
    )
    if answer == "Inference":
        st.success("Exactly. The model can answer without being shown the correct answer.")
    elif answer == "Training":
        st.info("Almost. Training needs the correct answer so the model can learn from it.")
    takeaway("First decide whether the model is answering or learning. Everything else will make more sense after that.")


def render_step_2() -> None:
    cat_stage()
    st.write("Imagine two helpers. Both look at exactly the same conceptual picture. They do not agree — and that is okay.")
    left, right = st.columns(2)
    with left:
        html_card("EXPERT 1", "“I think it is a cat.”", "Scientifically precise score: <b>o₁ = 0.90</b>", color="blue", big="90% cat")
    with right:
        html_card("EXPERT 2", "“I am not so sure.”", "Scientifically precise score: <b>o₂ = 0.20</b>", color="blue", big="20% cat")
    st.caption("Here, scores between 0 and 1 can be read intuitively as percentages. The implemented experts are linear, so a score is not guaranteed to stay in that range.")
    with st.expander("Where do these numbers come from?"):
        st.write("Each expert has its own weight **W** and bias **b**.")
        equation("A linear expert", "oᵢ(x) = xWᵢ + bᵢ")
        st.write("Our input is x = 1. Expert 1 has W = 0.90; Expert 2 has W = 0.20; both biases are 0. Same input. Different parameters. Different answers.")
    takeaway("Experts make predictions. They do NOT decide how much influence they get.")
    source_note("Teaching setup: two scalar linear experts. The paper uses networks with all experts receiving the same input.")


def render_step_3() -> None:
    st.write("The gate is like a teacher deciding how much to listen to each helper.")
    p1 = st.slider("Gate preference for Expert 1", 0.05, 0.95, 0.80, 0.01, key="demo_gate_probability")
    p = np.array([p1, 1.0 - p1])
    meters([("Expert 1", p[0], "blue"), ("Expert 2", p[1], "blue")])
    st.info(f"The weights add to 100%: {p[0]:.0%} + {p[1]:.0%} = 100%.")
    st.write("The gate has not seen the correct answer. This is only its initial preference.")
    with st.expander("How does the gate calculate these weights?"):
        equation("Raw gate scores", "z = xW_g + b_g")
        equation("Softmax", "pᵢ = exp(zᵢ / τ) / Σⱼ exp(zⱼ / τ)")
        delta = math.log(p1 / (1 - p1))
        st.write(f"At τ = 1, a two-logit difference of **{delta:.3f}** produces {p1:.0%} / {1-p1:.0%}. We derive a valid logit difference; we do not pretend arbitrary weights are already normalized.")
        tau = st.slider("Teaching extension: temperature τ", 0.4, 2.0, 1.0, 0.1, key="gate_tau_demo")
        fixed = np.exp(np.array([math.log(4), 0.0]) / tau)
        fixed /= fixed.sum()
        st.write(f"Keeping the original logits [ln(4), 0] fixed, τ = {tau:.1f} gives **{fixed[0]:.1%} / {fixed[1]:.1%}**. A lower temperature makes the larger logit more dominant.")
        st.caption("Temperature τ is an educational implementation extension. It is not part of the paper’s displayed softmax equation.")
    takeaway("The gate calculates weights. It has not made the final prediction yet.")


def render_step_4() -> None:
    st.write("The same 80% / 20% gate probabilities can be used in three different ways.")
    model = make_default_model()
    choice = st.radio(
        "Choose a prediction strategy",
        [
            "SOFT — Listen to BOTH experts.",
            "HARD — Listen only to the highest-weight expert.",
            "STOCHASTIC — Randomly choose one expert.",
        ],
        key="routing_mode",
    )
    if choice.startswith("SOFT"):
        soft_prediction = float(model.predict_soft(X)[0, 0])
        equation("Soft prediction", "0.80 × 0.90 + 0.20 × 0.20 = 0.76")
        html_card("FINAL CAT SCORE", "Both experts contributed.", "Expert 1 had more influence.", color="blue", big=f"{soft_prediction:.0%}")
    elif choice.startswith("HARD"):
        hard_prediction, hard_index = model.predict_hard(X, return_indices=True)
        equation("Hard deterministic prediction", "arg max(0.80, 0.20) → Expert 1 → 0.90")
        html_card("FINAL CAT SCORE", f"Only Expert {hard_index[0] + 1} was used.", "Its gate probability was the largest.", color="blue", big=f"{hard_prediction[0,0]:.0%}")
    else:
        if st.button("🎲 Roll the weighted die", key="roll_weighted_die", type="primary"):
            _, indices = model.predict_stochastic(X, rng=st.session_state.routing_rng, return_indices=True)
            st.session_state.stochastic_pick = int(indices[0])
        pick = st.session_state.stochastic_pick
        if pick is None:
            st.info("Press the button. Expert 1 has an 80% chance, but Expert 2 can still be chosen.")
        else:
            score = DEFAULT_OUTPUTS[pick]
            html_card("THE DIE CHOSE", f"Expert {pick + 1}", "No learning happened. This was only a prediction choice.", color="blue", big=f"score {score:.2f}")
    takeaway("These are three ways to turn gate probabilities into a prediction. They are NOT three different learning algorithms.")
    source_note("Figure 1 (paper p. 81) uses a conceptual stochastic selector; the paper also evaluates weighted combinations. Both are uses of the same mixture probabilities.")


def render_step_5() -> None:
    st.markdown('<div class="chapter-kicker">INFERENCE EXPERIMENT</div>', unsafe_allow_html=True)
    p1 = st.slider("How much should we listen to Expert 1?", 0.0, 1.0, 0.80, 0.01, key="soft_gate_demo")
    p2 = 1.0 - p1
    c1, c2 = p1 * 0.9, p2 * 0.2
    left, right = st.columns(2)
    with left:
        html_card("EXPERT 1 CONTRIBUTION", f"0.90 × {p1:.2f}", color="blue", big=f"{c1:.3f}")
    with right:
        html_card("EXPERT 2 CONTRIBUTION", f"0.20 × {p2:.2f}", color="blue", big=f"{c2:.3f}")
    equation("Add the contributions", f"{c1:.3f} + {c2:.3f} = {c1+c2:.3f}")
    st.info("No correct answer has been provided yet. No learning has happened. No expert or gate parameter was updated.")
    takeaway("Soft prediction simply listens to the experts in different proportions.")


def render_step_6() -> None:
    answer = st.radio(
        "With SOFT routing and weights of 80% / 20%, what happens?",
        [
            "Choose an answer…",
            "A. We choose Expert 1 only.",
            "B. We combine both experts, giving Expert 1 more influence.",
            "C. We randomly choose one expert.",
        ],
        key="quiz_inference",
    )
    if answer.startswith("B."):
        st.success("Yes. Soft prediction combines both answers, weighted 80% and 20%.")
    elif answer.startswith("A."):
        st.info("That describes hard deterministic routing. Soft routing keeps both experts.")
    elif answer.startswith("C."):
        st.info("That describes stochastic routing. Soft routing does not roll a die.")
    html_card("REMEMBER", "We now know how the model answers.", "Next, we will reveal the correct answer and see how the model learns.", color="blue", css_class="phase-card")
    takeaway("Inference produces an answer without changing the model.")


def render_step_7() -> None:
    cat_stage("The same picture, experts, and gate — but now we are training.")
    st.write("Expert 1 still says **0.90**. Expert 2 still says **0.20**. The gate still says **80% / 20%**.")
    if st.button("Reveal the correct answer", key="reveal_target", type="primary"):
        st.session_state.target_revealed = True
    if st.session_state.target_revealed:
        html_card("TRAINING TARGET", "YES — it really is a cat.", "The scalar target is <b>y = 1.00</b>.", color="green", big="1.00", css_class="phase-card")
        st.write("During inference, we did not need this answer. During training, we do. Now we can measure how wrong each expert was.")
    else:
        st.info("During inference this label was hidden. Reveal it to begin training.")
    takeaway("The correct target is what allows us to learn.")


def render_step_8() -> None:
    st.write("The target is 1.00. We compare each helper with that target separately.")
    left, right = st.columns(2)
    with left:
        html_card("EXPERT 1", "Predicted 0.90", "(1.00 − 0.90)² = <b>0.01</b>", color="green", big="error 0.01")
    with right:
        html_card("EXPERT 2", "Predicted 0.20", "(1.00 − 0.20)² = <b>0.64</b>", color="orange", big="error 0.64")
    meters([("Expert 1 error", 0.01, "green"), ("Expert 2 error", 0.64, "orange")], maximum=0.64)
    st.write("Expert 1 was much closer. Expert 2 was much further away. That only describes this one example; it does not prove Expert 2 is useless.")
    takeaway("Each expert can be evaluated separately.")


def render_step_9() -> None:
    st.write("An earlier cooperative approach first blended the experts. Only then did it check the error.")
    equation("Paper Equation (1.1)", "E_coop = ‖y − Σᵢ pᵢoᵢ‖²")
    equation("Our numbers", "combined = 0.80(0.90) + 0.20(0.20) = 0.76")
    equation("Then check the combined error", "E_coop = (1.00 − 0.76)² = 0.0576")
    html_card("OLD TRAINING ERROR", "Error after combining", "This objective is not always bad. But it can let experts compensate for one another when we want them to specialize.", color="orange", big="0.0576")
    takeaway("Equation 1.1 judges the blended answer, not each expert’s answer separately.")
    source_note("Jacobs et al. (1991), Equation 1.1, paper p. 80.")


def render_step_10() -> None:
    st.write("Keep the target at 1.00, Expert 1 at 0.90, and the gate at 80% / 20%. Move only Expert 2’s score.")
    o2 = st.slider("Change Expert 2’s output score", 0.0, 1.6, 0.20, 0.01, key="counterexample_o2")
    combined = 0.8 * 0.9 + 0.2 * o2
    error = (1.0 - combined) ** 2
    c1, c2, c3 = st.columns(3)
    c1.metric("Expert 1 score", "0.900")
    c2.metric("Expert 2 score", f"{o2:.3f}")
    c3.metric("Combined prediction", f"{combined:.3f}")
    equation("Cooperative error", f"(1.00 − {combined:.3f})² = {error:.6f}")
    st.button(
        "Show how two wrong experts can give a perfect combined answer",
        key="special_button_counterexample",
        on_click=set_counterexample,
    )
    if math.isclose(o2, 1.4, abs_tol=1e-9):
        st.success("The combined answer is exactly 1.00, so Equation 1.1 says the error is zero.")
        warning("Wait a minute! Expert 1 says 0.90. Expert 2 says 1.40. Neither says exactly 1.00. Their errors merely cancel.")
        st.write("**1.40 is a raw linear score, not a 140% probability.** A true probability cannot exceed 100%. We temporarily allow unrestricted scores to reveal the mathematical issue.")
    takeaway("The final blended answer can be correct even when individual experts are wrong.")
    source_note("This is an isolated thought experiment. It does not modify the trainable cat model used later.")


def render_step_11() -> None:
    st.write("Instead of checking only the blend, ask: “How wrong is each expert?” Then weight those separate errors.")
    equation("Paper Equation (1.2)", "E_expected = Σᵢ pᵢ(y − oᵢ)²")
    equation("Default cat example", "0.80(0.01) + 0.20(0.64) = 0.008 + 0.128 = 0.136")
    st.markdown("#### Now revisit the cancellation case")
    left, right = st.columns(2)
    with left:
        html_card("EQUATION 1.1", "Check only the blend", "0.8(0.9) + 0.2(1.4) = 1.0", color="orange", big="E = 0")
    with right:
        html_card("EQUATION 1.2", "Check individual errors", "0.8(0.01) + 0.2(0.16)", color="green", big="E = 0.04")
    st.write("The old error says the combined answer is perfect. Equation 1.2 still notices that the individual experts made mistakes.")
    with st.expander("Advanced: a useful identity"):
        st.write("Expected individual error = error of the weighted mean + gate-weighted variation among expert predictions. The extra variation term is why disagreement cannot simply disappear through cancellation.")
    takeaway("Judging experts separately exposes mistakes that a blended answer can hide.")
    source_note("Jacobs et al. (1991), Equation 1.2, paper p. 80.")


def render_step_12() -> None:
    st.write("Imagine rolling a weighted die. Expert 1 is chosen 80% of the time. Expert 2 is chosen 20% of the time.")
    col1, col2 = st.columns(2)
    count = None
    if col1.button("Simulate 100 selections", key="simulate_100"):
        count = 100
    if col2.button("Simulate 1,000 selections", key="simulate_1000"):
        count = 1000
    if count:
        picks = st.session_state.simulation_rng.choice(2, size=count, p=DEFAULT_GATE)
        counts = np.bincount(picks, minlength=2)
        observed = float(np.mean(np.where(picks == 0, 0.01, 0.64)))
        st.session_state.simulation_result = (count, counts, observed)
    if st.session_state.simulation_result:
        count, counts, observed = st.session_state.simulation_result
        c1, c2, c3 = st.columns(3)
        c1.metric("Expert 1 selected", f"{counts[0]}×")
        c2.metric("Expert 2 selected", f"{counts[1]}×")
        c3.metric("Observed avg. error", f"{observed:.4f}")
        st.write(f"Exact expected error: **0.1360**. The {count:,}-roll simulation is an approximation.")
    equation("No rolls needed", "0.80(0.01) + 0.20(0.64) = 0.136")
    st.write("Equation 1.2 calculates the long-run average directly. We do not have to physically sample an expert during training.")
    takeaway("Stochastic selection is one interpretation. Expected error can be calculated without random sampling.")
    source_note("The selector in paper Figure 1 (p. 81) provides this stochastic interpretation.")


def render_step_13() -> None:
    st.write("Equation 1.2 was already an improvement. The authors then used a likelihood-based error that worked better in their experiments.")
    html_card("A GENTLE DEFINITION", "What is a likelihood?", "A number describing how well a possible explanation matches the answer we actually saw. A closer expert has a better fit.", color="blue", css_class="phase-card")
    equation("Paper Equation (1.3)", "L_paper = −log[Σᵢ pᵢ exp(−(y − oᵢ)² / 2)]")
    st.write("Do not memorize it. It does three things: checks each expert’s fit, combines those fits using the gate, then turns the result into a loss we minimize.")
    equation("Teaching extension", "L = −log[Σᵢ pᵢ exp(−(y − oᵢ)² / (2σ²))]")
    st.info("We added σ so you can explore how strongly the model reacts to error differences. Setting σ = 1 gives the paper’s displayed Gaussian scale.")
    takeaway("This loss is built from individual expert fits. It is not the squared error of the weighted-average prediction.")
    source_note("Jacobs et al. (1991), Equation 1.3, paper p. 82. Adjustable σ is this simulator’s teaching extension.")


def render_step_14() -> None:
    sigma = st.slider("Gaussian error scale σ", 0.3, 2.0, 0.5, 0.1, key="likelihood_sigma")
    errors = np.array([0.01, 0.64])
    fits = np.exp(-errors / (2 * sigma**2))
    weighted = DEFAULT_GATE * fits
    mixture_fit = float(weighted.sum())
    loss = -math.log(mixture_fit)
    st.markdown("**1 · Find each squared error**")
    equation("Individual errors", "d₁ = (1−0.9)² = 0.01    ·    d₂ = (1−0.2)² = 0.64")
    st.markdown("**2 · Turn error into fit** — smaller error means larger fit.")
    equation("Fit values", f"fit₁ = exp(−0.01/(2×{sigma:.1f}²)) = {fits[0]:.4f}    ·    fit₂ = {fits[1]:.4f}")
    st.markdown("**3 · Multiply by the gate probabilities**")
    equation("Weighted fits", f"0.8×{fits[0]:.4f} = {weighted[0]:.4f}    ·    0.2×{fits[1]:.4f} = {weighted[1]:.4f}")
    st.markdown("**4 · Add, then take the negative logarithm**")
    equation("Mixture loss", f"L = −log({mixture_fit:.4f}) = {loss:.4f}")
    html_card("COMPETITIVE MIXTURE LOSS", f"At σ = {sigma:.1f}", "Computed from unrounded internal values.", color="green", big=f"{loss:.4f}")
    with st.expander("What do exp and log mean here?"):
        st.write("**exp** turns a small error into a fit near 1 and a large error into a smaller fit. **−log** turns the total fit into a loss: better total fit means a smaller loss.")
    warning("Changing σ changes the objective’s sensitivity. Loss values from different σ settings are not directly comparable as though they were the same objective.")
    takeaway("At the default σ = 0.5, the loss is about 0.1746.")


def render_step_15() -> None:
    st.write("The blue numbers are the gate’s guess before seeing the target. The green numbers assign credit after seeing it.")
    sigma = st.slider("Error scale σ", 0.05, 1.5, 0.5, 0.05, key="responsibility_sigma")
    o2 = st.slider("Expert 2 score in this isolated experiment", 0.0, 1.2, 0.2, 0.02, key="responsibility_o2")
    outputs = np.array([0.9, o2])
    errors = (1.0 - outputs) ** 2
    terms = DEFAULT_GATE * np.exp(-errors / (2 * sigma**2))
    h = terms / terms.sum()
    left, right = st.columns(2)
    with left:
        html_card("BEFORE THE ANSWER", "Gating probability p", "Input-only preference", color="blue", css_class="phase-card")
        meters([("Expert 1", 0.8, "blue"), ("Expert 2", 0.2, "blue")])
    with right:
        html_card("AFTER THE ANSWER", "Responsibility h", "Target-aware training credit", color="green", css_class="phase-card")
        meters([("Expert 1", h[0], "green"), ("Expert 2", h[1], "green")])
    equation("Posterior responsibility", "hᵢ = pᵢ exp(−dᵢ/(2σ²)) / Σⱼ pⱼ exp(−dⱼ/(2σ²))")
    st.markdown("**Responsibility = normalize(gating weight × how well the expert fits).**")
    if h[1] > h[0]:
        st.success("Look! Expert 2 began with the lower gate probability, yet its much better fit earned more responsibility.")
    st.info("Responsibilities are calculated during training because they need the correct target. Ordinary inference does not need them.")
    takeaway("Gating probability asks what the gate expects. Responsibility asks who best explained the target we actually saw.")


def render_step_16() -> None:
    model = make_default_model()
    h = model.compute_responsibilities(X, Y)[0]
    st.write("A gradient is simply a mathematical instruction for which direction to change something.")
    equation("Paper Equation (1.4) · expected-error gradient", "∂E_expected/∂oᵢ = 2pᵢ(oᵢ − y)")
    equation("Paper Equation (1.5) · mixture-likelihood gradient", "∂L_paper/∂oᵢ = hᵢ(oᵢ − y)")
    left, right = st.columns(2)
    with left:
        html_card("OLD CREDIT FACTOR", "Gate probability p", "80.0% / 20.0%", color="blue", css_class="phase-card")
        meters([("Expert 1", 0.8, "blue"), ("Expert 2", 0.2, "blue")])
    with right:
        html_card("NEW CREDIT FACTOR", "Responsibility h", f"{h[0]:.1%} / {h[1]:.1%}", color="green", css_class="phase-card")
        meters([("Expert 1", h[0], "green"), ("Expert 2", h[1], "green")])
    warning("These formulas belong to different losses and have different scale factors. A larger responsibility alone does not guarantee a larger entire numerical gradient; the expert’s error also matters.")
    with st.expander("Advanced: our teaching extension with σ = 0.5"):
        equation("Scaled expert gradient", "∂L/∂oᵢ = hᵢ(oᵢ − y) / σ²")
        gradients = h * (DEFAULT_OUTPUTS - 1.0) / 0.5**2
        st.write(f"Expert 1: **{gradients[0]:.8f}**. Expert 2: **{gradients[1]:.8f}**.")
    takeaway("The mixture likelihood allocates training credit using both gate preference and individual fit.")
    source_note("Jacobs et al. (1991), Equations 1.4 and 1.5, paper p. 82.")


def render_step_17() -> None:
    st.write("The experts contain weights and biases. Training changes those numbers a little, so future predictions change.")
    equation("Each expert", "o = xW + b")
    left, right = st.columns(2)
    with left:
        html_card("EXPERT 1", "x = 1, W = 0.90, b = 0", "1(0.90) + 0 = 0.90", color="blue", big="o₁ = 0.90")
    with right:
        html_card("EXPERT 2", "x = 1, W = 0.20, b = 0", "1(0.20) + 0 = 0.20", color="blue", big="o₂ = 0.20")
    equation("The gate", "z = xW_g + b_g    ·    p = softmax(z/τ)")
    st.write("The gate starts with **W_g = [ln(4), 0]** and **b_g = [0, 0]**, so at x = 1 and τ = 1 it produces **[0.8, 0.2]**.")
    eta = st.slider(
        "Learning rate η — how big should each step be?",
        0.01,
        0.20,
        value=float(st.session_state.get("selected_eta", 0.10)),
        step=0.01,
        key="main_eta_widget",
        on_change=remember_learning_rate,
    )
    st.info(f"The next demonstrations will use η = {eta:.2f}. Changing this really changes the parameter updates.")
    takeaway("Training changes weights and biases in a direction intended to reduce the loss.")


def render_step_18() -> None:
    eta = st.session_state.get("selected_eta", 0.10)
    model, result = default_training_preview(eta)
    h = result["gradients"]["responsibilities"][0]
    grads = result["gradients"]["expert_output_grads"][0, :, 0]
    after = result["after"]["expert_outputs"][0, :, 0]
    st.write("Both experts predicted below the correct answer. Their gradients are negative, so gradient descent pushes their outputs upward.")
    equation("Competitive expert gradient", "∂L/∂oᵢ = hᵢ(oᵢ − y) / σ²")
    left, right = st.columns(2)
    with left:
        html_card("EXPERT 1", f"h = {h[0]:.4f}, output gradient = {grads[0]:.8f}", "Before 0.9000", color="green", big=f"→ {after[0]:.4f}")
    with right:
        html_card("EXPERT 2", f"h = {h[1]:.4f}, output gradient = {grads[1]:.8f}", "Before 0.2000", color="green", big=f"→ {after[1]:.4f}")
    with st.expander("Show how the weights changed"):
        for i, expert in enumerate(model.experts):
            grad_w = result["gradients"]["expert_weight_grads"][i][0, 0]
            grad_b = result["gradients"]["expert_bias_grads"][i][0]
            st.markdown(f"**Expert {i+1}**")
            st.code(
                f"W: {DEFAULT_OUTPUTS[i]:.8f} - {eta:.2f} × ({grad_w:.8f}) = {expert.weight[0,0]:.8f}\n"
                f"b: 0.00000000 - {eta:.2f} × ({grad_b:.8f}) = {expert.bias[0]:.8f}\n"
                f"new output: 1 × W + b = {after[i]:.8f}"
            )
    st.write("Because x = 1, both W and b affect the recomputed output equally. Its total change contains both parameter updates. There is **no extra factor of 2** in the mixture-loss output gradient.")
    takeaway("Responsibility matters, but update size also depends on how far the expert is from the target.")


def render_step_19() -> None:
    eta = st.session_state.get("selected_eta", 0.10)
    _, result = default_training_preview(eta)
    h = result["gradients"]["responsibilities"][0]
    grad_z = result["gradients"]["gate_logit_grads"][0]
    new_p = result["after"]["gate_probs"][0]
    st.write("The gate predicted 80% / 20%. The target-aware responsibilities are about 93.4% / 6.6%. That difference is the gate’s learning signal.")
    equation("Gradient of raw gate logits", "∂L/∂zᵢ = (pᵢ − hᵢ) / τ")
    equation("At τ = 1", f"grad z₁ = {grad_z[0]:+.8f}    ·    grad z₂ = {grad_z[1]:+.8f}")
    st.markdown("**Before**")
    meters([("Expert 1", 0.8, "blue"), ("Expert 2", 0.2, "blue")])
    st.markdown("**After updating both gate weights and biases, then applying softmax again**")
    meters([("Expert 1", new_p[0], "green"), ("Expert 2", new_p[1], "green")])
    st.info(f"At η = {eta:.2f}, the gate moves to {new_p[0]:.4%} / {new_p[1]:.4%}. It does not jump directly to the responsibilities.")
    takeaway("Responsibilities teach the gate. They are not copied into the gate as its new probabilities.")


def _ensure_learning_demo() -> None:
    eta = float(st.session_state.get("selected_eta", 0.10))
    signature = (eta,)
    if st.session_state.get("learning_signature") != signature:
        model = make_default_model(learning_rate=eta)
        initial_loss = model.compute_competitive_loss(X, Y)
        first = model.train_step(X, Y)
        st.session_state.teaching_model = model
        st.session_state.learning_signature = signature
        st.session_state.teaching_training_step_count = 1
        st.session_state.training_history = [
            {"step": 0, "prediction": 0.76, "loss": initial_loss},
            {"step": 1, "prediction": float(first["soft_prediction_after"][0, 0]), "loss": first["after_loss"]},
        ]


def _reset_learning_demo() -> None:
    for key in ("learning_signature", "teaching_model", "teaching_training_step_count", "training_history"):
        st.session_state.pop(key, None)


def render_step_20() -> None:
    _ensure_learning_demo()
    model = st.session_state.teaching_model
    if st.button("Run another training update", key="run_another_update", type="primary"):
        result = model.train_step(X, Y)
        st.session_state.teaching_training_step_count += 1
        st.session_state.training_history.append(
            {
                "step": st.session_state.teaching_training_step_count,
                "prediction": float(result["soft_prediction_after"][0, 0]),
                "loss": result["after_loss"],
            }
        )
    current = model.forward(X)
    current_outputs = current["expert_outputs"][0, :, 0]
    current_gate = current["gate_probs"][0]
    current_prediction = float(model.predict_soft(X)[0, 0])
    current_loss = model.compute_competitive_loss(X, Y)
    left, right = st.columns(2)
    with left:
        html_card("BEFORE LEARNING", "Initial model", "Experts: 0.9000 / 0.2000<br>Gate: 0.8000 / 0.2000", color="blue", big="soft = 0.7600")
    with right:
        html_card(
            f"AFTER {st.session_state.teaching_training_step_count} UPDATE(S)",
            "Same cat input",
            f"Experts: {current_outputs[0]:.4f} / {current_outputs[1]:.4f}<br>Gate: {current_gate[0]:.4f} / {current_gate[1]:.4f}",
            color="green",
            big=f"soft = {current_prediction:.4f}",
        )
    c1, c2, c3 = st.columns(3)
    c1.metric("Correct target", "1.0000")
    c2.metric("Current loss", f"{current_loss:.6f}")
    c3.metric("Update number", st.session_state.teaching_training_step_count)
    history_chart(st.session_state.training_history)
    if len(st.session_state.training_history) > 1 and st.session_state.training_history[-1]["loss"] > st.session_state.training_history[-2]["loss"]:
        warning("This step increased the loss. Gradient descent can overshoot when a learning rate is too large; improvement is not hardcoded.")
    if st.button("Reset Learning", key="reset_learning", on_click=_reset_learning_demo):
        st.rerun()
    st.write("At the default settings, the first update changes the soft prediction from **0.7600** to about **0.8344**, and the competitive loss from **0.1746** to about **0.1414**.")
    takeaway("The experts changed. The gate changed. So a new forward pass gives a new answer. That is learning.")


SPECIAL_X = np.array([[-1.5], [-1.2], [-0.8], [-0.5], [0.5], [0.8], [1.2], [1.5]])
SPECIAL_Y = np.array([[0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0]])


def _specialization_row(model, epoch: int) -> dict[str, float]:
    probes = np.array([[1.0], [-1.0]])
    forward = model.forward(probes)
    return {
        "step": epoch,
        "gate_cat": float(forward["gate_probs"][0, 0]),
        "gate_no_cat": float(forward["gate_probs"][1, 0]),
    }


def _ensure_specialization_demo() -> None:
    if "mini_experiment_model" not in st.session_state:
        model = make_specialization_model(seed=1991)
        st.session_state.mini_experiment_model = model
        st.session_state.mini_experiment_epoch = 0
        st.session_state.mini_experiment_history = [_specialization_row(model, 0)]


def _reset_specialization() -> None:
    for key in ("mini_experiment_model", "mini_experiment_epoch", "mini_experiment_history"):
        st.session_state.pop(key, None)


def render_step_21() -> None:
    st.write("One cat example explains one update. Real learning uses many examples, so different helpers can become useful in different input regions.")
    st.info("This is a tiny, artificially constructed scalar dataset: negative x means a no-cat case (target 0); positive x means a clear-cat case (target 1). It is not image recognition.")
    _ensure_specialization_demo()
    model = st.session_state.mini_experiment_model
    c1, c2, c3 = st.columns(3)
    epochs = 0
    if c1.button("Train 1 epoch", key="train_one_epoch"):
        epochs = 1
    if c2.button("Train 10 epochs", key="train_ten_epochs"):
        epochs = 10
    if c3.button("Reset experiment", key="reset_mini", on_click=_reset_specialization):
        st.rerun()
    for _ in range(epochs):
        model.train_step(SPECIAL_X, SPECIAL_Y)
        st.session_state.mini_experiment_epoch += 1
        st.session_state.mini_experiment_history.append(
            _specialization_row(model, st.session_state.mini_experiment_epoch)
        )
    probes = np.array([[1.0], [-1.0]])
    forward = model.forward(probes)
    predictions = model.predict_soft(probes)[:, 0]
    left, right = st.columns(2)
    with left:
        html_card("CLEAR CAT REGION", "x = +1, target = 1", f"Expert scores: {forward['expert_outputs'][0,0,0]:.3f} / {forward['expert_outputs'][0,1,0]:.3f}<br>Gate: {forward['gate_probs'][0,0]:.1%} / {forward['gate_probs'][0,1]:.1%}", color="green", big=f"soft {predictions[0]:.3f}")
    with right:
        html_card("NO-CAT REGION", "x = −1, target = 0", f"Expert scores: {forward['expert_outputs'][1,0,0]:.3f} / {forward['expert_outputs'][1,1,0]:.3f}<br>Gate: {forward['gate_probs'][1,0]:.1%} / {forward['gate_probs'][1,1]:.1%}", color="blue", big=f"soft {predictions[1]:.3f}")
    history_chart(st.session_state.mini_experiment_history, specialization=True)
    st.caption(f"Deterministic seed 1991 · epoch {st.session_state.mini_experiment_epoch}. Specialization is an outcome to inspect, not a guaranteed animation.")
    with st.expander("Important: two different meanings of soft and hard"):
        st.write("**Prediction routing:** soft combines experts; hard uses the highest gate probability; stochastic samples one.")
        st.write("**Competitive learning:** soft training gives fractional posterior credit h to experts. Hard competitive learning approximates that with one target-aware best-fitting assignment.")
        warning("A hard training assignment depends on the known target and fit. It is not automatically the expert with the highest input-only gate probability.")
    takeaway("Across many examples, experts may become useful in different regions, while the gate learns when to favor each one. That is the idea of local experts.")
    source_note("Paper Section 2 distinguishes soft and hard competitive learning. Figures 2–3 show specialization for vowel recognition; this cat/no-cat scalar dataset is only a teaching analogy.")


def render_step_22() -> None:
    st.markdown(
        '<div class="flow-card blue"><div class="card-label">INFERENCE · ANSWERING</div>'
        '<div class="flow-line">Cat input <span class="arrow">→</span> experts predict <span class="arrow">→</span> gate gives probabilities <span class="arrow">→</span> soft weighted combination <span class="arrow">→</span> final answer</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="flow-card green"><div class="card-label">TRAINING · LEARNING</div>'
        '<div class="flow-line">Cat input + correct target <span class="arrow">→</span> predictions and gate probabilities <span class="arrow">→</span> individual errors <span class="arrow">→</span> mixture likelihood <span class="arrow">→</span> responsibilities <span class="arrow">→</span> expert + gate gradients <span class="arrow">→</span> update weights and biases <span class="arrow">→</span> try again</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown("**The experts make different predictions.**  \n**The gate decides how much influence each expert receives.**  \n**Training uses correct answers to teach both the experts and the gate.**")
    st.info("Prediction may combine experts softly, while learning uses a competitive mixture likelihood.")
    st.markdown("#### Quick final check")
    questions = [
        ("Does inference use the correct target?", ["Choose…", "No", "Yes"], "No"),
        ("Do gating probabilities and responsibilities mean the same thing?", ["Choose…", "No", "Yes"], "No"),
        ("Does hard routing mean the same thing as hard competitive learning?", ["Choose…", "No", "Yes"], "No"),
        ("Does competitive loss simply square the error of the combined prediction?", ["Choose…", "No", "Yes"], "No"),
        ("Do weights and biases change during inference?", ["Choose…", "No", "Yes"], "No"),
    ]
    correct = 0
    answered = 0
    for index, (prompt, options, expected) in enumerate(questions):
        answer = st.radio(prompt, options, key=f"final_quiz_{index}", horizontal=True)
        if answer != "Choose…":
            answered += 1
            if answer == expected:
                correct += 1
                st.caption("✓ Exactly.")
            else:
                st.caption("Not quite. Look back at the two flows above: inference has no target or updates, and the paired terms have different jobs.")
    if answered == len(questions):
        if correct == len(questions):
            st.success("Wonderful — all five answers are correct.")
        else:
            st.info(f"You have {correct} of {len(questions)}. Try the remaining idea again gently; nothing is timed.")
    takeaway("You have completed the Adaptive Mixture of Local Experts walkthrough.", "You made it")
    source_note(f"{PAPER_CITATION} The original experiment used vowel recognition. This tutorial does not reproduce those results.")


RENDERERS = [
    render_step_1,
    render_step_2,
    render_step_3,
    render_step_4,
    render_step_5,
    render_step_6,
    render_step_7,
    render_step_8,
    render_step_9,
    render_step_10,
    render_step_11,
    render_step_12,
    render_step_13,
    render_step_14,
    render_step_15,
    render_step_16,
    render_step_17,
    render_step_18,
    render_step_19,
    render_step_20,
    render_step_21,
    render_step_22,
]


def render_navigation(step_index: int) -> None:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    previous, reset, next_column = st.columns([1, 1, 1])
    previous.button(
        "← Previous",
        key="nav_previous",
        on_click=previous_step,
        disabled=step_index == 0,
        width="stretch",
    )
    reset.button("Reset", key="nav_reset", on_click=reset_lesson, width="stretch")
    next_column.button(
        "Next →",
        key="nav_next",
        on_click=next_step,
        disabled=step_index == TOTAL_STEPS - 1,
        type="primary",
        width="stretch",
    )


def main() -> None:
    initialize_state()
    step_index = int(np.clip(st.session_state.lesson_step, 0, TOTAL_STEPS - 1))
    st.session_state.lesson_step = step_index
    step = LESSON_STEPS[step_index]
    with st.container(key="lesson_card"):
        st.markdown(
            f'<div class="chapter-kicker">{step.chapter} · STEP {step_index + 1} OF {TOTAL_STEPS}</div>'
            f'<div class="chapter-name">{step.chapter_title}</div>',
            unsafe_allow_html=True,
        )
        st.progress((step_index + 1) / TOTAL_STEPS)
        st.title(step.title)
        RENDERERS[step_index]()
        render_navigation(step_index)


if __name__ == "__main__":
    main()
