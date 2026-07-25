# -*- coding: utf-8 -*-
"""The probability-versus-volume plot must use one colour and no colour bar.

Reviewer 1 asked what the blue-to-yellow dots on Fig 3 encoded. They encoded
artifact probability, which the vertical axis already carries, so the colour
scale duplicated that axis and could be read as a second variable. The reply
states that it was removed and that all points are drawn in a single colour, and
the Fig 3 caption says the same, so the code that a reader runs has to agree.
"""
import ast
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "src" / "ml_threshold_selection" / "ui_visualization.py"


def _scatter_calls(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = getattr(func, "attr", None) or getattr(func, "id", None)
            if name == "scatter":
                yield node


def test_no_scatter_maps_a_variable_onto_colour():
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    offenders = []
    for call in _scatter_calls(tree):
        keywords = {k.arg for k in call.keywords if k.arg}
        if "cmap" in keywords or "c" in keywords:
            offenders.append(call.lineno)
    assert not offenders, (
        f"scatter at line(s) {offenders} passes a colour variable. The article "
        "plots these points in a single colour."
    )


def test_no_colour_bar_is_drawn():
    source = MODULE.read_text(encoding="utf-8")
    assert "colorbar" not in source, (
        "a colour bar is drawn, but the points carry no colour variable"
    )


def test_every_scatter_sets_an_explicit_single_colour():
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    calls = list(_scatter_calls(tree))
    assert calls, "expected at least one scatter call"
    for call in calls:
        keywords = {k.arg for k in call.keywords if k.arg}
        assert "color" in keywords, (
            f"scatter at line {call.lineno} does not set an explicit colour"
        )


def test_the_shared_colour_constant_is_a_grey():
    import re
    source = MODULE.read_text(encoding="utf-8")
    match = re.search(r'SCATTER_GREY\s*=\s*"(#[0-9a-fA-F]{6})"', source)
    assert match, "SCATTER_GREY is not defined"
    r, g, b = (int(match.group(1)[i:i + 2], 16) for i in (1, 3, 5))
    assert r == g == b, f"{match.group(1)} is not a neutral grey"
