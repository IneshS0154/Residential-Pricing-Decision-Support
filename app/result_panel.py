"""Result panel as ONE persistent Streamlit component (plain HTML/JS, no build step).

Streamlit reruns the script on every input change and, for native charts, replaces the element. A component with a stable key keeps its iframe,
so the page receives new numbers and tweens from the old values to the new ones (price, interval bar, gauge, waterfall bars) instead of redrawing.
"""
from pathlib import Path

import streamlit.components.v1 as components

_comp = components.declare_component("result_panel", path=str(Path(__file__).parent / "components" / "result_panel"))


def result_panel(data, key="result_panel"):
    return _comp(data=data, key=key, default=None)
