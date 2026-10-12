import pytest
from streamlit.testing.v1 import AppTest

from lesson_content import LESSON_STEPS


APP_PATH = "../app.py"


@pytest.mark.parametrize("step_index", range(22))
def test_every_lesson_step_renders_without_exceptions(step_index):
    app = AppTest.from_file(APP_PATH)
    app.session_state["lesson_step"] = step_index
    app.run(timeout=25)
    assert not app.exception
    assert app.title[0].value == LESSON_STEPS[step_index].title


def test_next_and_previous_navigation_changes_exactly_one_step():
    app = AppTest.from_file(APP_PATH).run(timeout=25)
    assert app.session_state["lesson_step"] == 0

    app.button(key="nav_next").click().run(timeout=25)
    assert app.session_state["lesson_step"] == 1
    assert app.title[0].value == LESSON_STEPS[1].title

    app.button(key="nav_previous").click().run(timeout=25)
    assert app.session_state["lesson_step"] == 0
    assert app.title[0].value == LESSON_STEPS[0].title


def test_navigation_boundaries_are_disabled():
    first = AppTest.from_file(APP_PATH).run(timeout=25)
    assert first.button(key="nav_previous").disabled

    last = AppTest.from_file(APP_PATH)
    last.session_state["lesson_step"] = 21
    last.run(timeout=25)
    assert last.button(key="nav_next").disabled


def test_reset_restarts_lesson_and_clears_modified_state():
    app = AppTest.from_file(APP_PATH)
    app.session_state["lesson_step"] = 9
    app.run(timeout=25)
    app.slider(key="counterexample_o2").set_value(1.4).run(timeout=25)
    assert app.slider(key="counterexample_o2").value == pytest.approx(1.4)

    app.button(key="nav_reset").click().run(timeout=25)
    assert app.session_state["lesson_step"] == 0
    assert app.title[0].value == LESSON_STEPS[0].title
    assert "counterexample_o2" not in app.session_state


def test_real_additional_training_button_changes_model_state():
    app = AppTest.from_file(APP_PATH)
    app.session_state["lesson_step"] = 19
    app.run(timeout=25)
    assert app.session_state["teaching_training_step_count"] == 1
    before = app.session_state["teaching_model"].predict_soft([[1.0]])[0, 0]

    app.button(key="run_another_update").click().run(timeout=25)
    after = app.session_state["teaching_model"].predict_soft([[1.0]])[0, 0]
    assert app.session_state["teaching_training_step_count"] == 2
    assert after != pytest.approx(before)


def test_learning_rate_selection_survives_navigation_to_update_steps():
    app = AppTest.from_file(APP_PATH)
    app.session_state["lesson_step"] = 16
    app.run(timeout=25)
    app.slider(key="main_eta_widget").set_value(0.2).run(timeout=25)
    assert app.session_state["selected_eta"] == pytest.approx(0.2)

    app.button(key="nav_next").click().run(timeout=25)
    assert app.session_state["lesson_step"] == 17
    assert app.session_state["selected_eta"] == pytest.approx(0.2)
