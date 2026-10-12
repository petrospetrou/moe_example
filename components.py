"""Small visual helpers for the Streamlit lesson."""

from __future__ import annotations

from html import escape
from typing import Iterable

import plotly.graph_objects as go
import streamlit as st


def cat_stage(subtitle: str = "A conceptual input — no real image recognition is happening.") -> None:
    st.markdown(
        f"""<div class="cat-stage"><div class="cat-emoji">🐈</div><div>
        <div class="cat-question">Is there a cat in this picture?</div>
        <div class="cat-sub">{escape(subtitle)}</div></div></div>""",
        unsafe_allow_html=True,
    )


def html_card(
    label: str,
    title: str,
    body: str = "",
    *,
    color: str = "",
    big: str | None = None,
    css_class: str = "expert-card",
) -> None:
    color_class = f" {color}" if color else ""
    big_html = f'<div class="big-number {color}-text">{escape(big)}</div>' if big else ""
    st.markdown(
        f"""<div class="{css_class}{color_class}">
        <div class="card-label">{escape(label)}</div>
        <div class="card-title">{escape(title)}</div>{big_html}
        <div class="small-copy">{body}</div></div>""",
        unsafe_allow_html=True,
    )


def equation(label: str, content: str) -> None:
    st.markdown(
        f'<div class="equation-box"><div class="equation-label">{escape(label)}</div>'
        f'<div class="equation">{escape(content)}</div></div>',
        unsafe_allow_html=True,
    )


def meters(rows: Iterable[tuple[str, float, str]], *, maximum: float = 1.0) -> None:
    markup = []
    maximum = maximum if maximum > 0 else 1.0
    for label, value, color in rows:
        width = max(0.0, min(100.0, float(value) / maximum * 100.0))
        display = f"{value:.1%}" if maximum <= 1.0 else f"{value:.4f}"
        markup.append(
            f'<div class="meter-row"><div class="meter-label">{escape(label)}</div>'
            f'<div class="meter-track"><div class="meter-fill {escape(color)}" style="width:{width:.3f}%"></div></div>'
            f'<div class="meter-value">{display}</div></div>'
        )
    st.markdown("".join(markup), unsafe_allow_html=True)


def takeaway(text: str, title: str = "What just happened?") -> None:
    st.markdown(
        f'<div class="takeaway"><div class="takeaway-title">{escape(title)}</div>'
        f'<div class="takeaway-copy">{escape(text)}</div></div>',
        unsafe_allow_html=True,
    )


def source_note(text: str) -> None:
    st.markdown(f'<div class="source-note">Source note · {escape(text)}</div>', unsafe_allow_html=True)


def warning(text: str) -> None:
    st.markdown(f'<div class="warning-box">{escape(text)}</div>', unsafe_allow_html=True)


def history_chart(history: list[dict[str, float]], *, specialization: bool = False) -> None:
    if not history:
        return
    x = [row["step"] for row in history]
    figure = go.Figure()
    if specialization:
        for key, label, color in (
            ("gate_cat", "Gate → Expert 1 on cat", "#5fa8ff"),
            ("gate_no_cat", "Gate → Expert 1 on no-cat", "#9eabc0"),
        ):
            figure.add_trace(go.Scatter(x=x, y=[r[key] for r in history], name=label, mode="lines+markers", line={"color": color}))
        y_title = "Gate probability"
    else:
        for key, label, color in (
            ("prediction", "Soft prediction", "#5fa8ff"),
            ("loss", "Competitive loss", "#55d6a5"),
        ):
            figure.add_trace(go.Scatter(x=x, y=[r[key] for r in history], name=label, mode="lines+markers", line={"color": color}))
        y_title = "Value"
    figure.update_layout(
        height=280,
        margin={"l": 12, "r": 12, "t": 25, "b": 12},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(9,15,24,.55)",
        font={"color": "#b9c5d7", "size": 12},
        legend={"orientation": "h", "y": 1.12, "x": 0},
        xaxis={"title": "Training update", "gridcolor": "#253247"},
        yaxis={"title": y_title, "gridcolor": "#253247"},
        hovermode="x unified",
    )
    st.plotly_chart(figure, width="stretch", config={"displayModeBar": False})
