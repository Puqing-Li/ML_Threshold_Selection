# -*- coding: utf-8 -*-
"""Every app attribute the modules read must actually exist on the app.

`app.voxel_sizes` was read in two places but never assigned; the attribute is
`training_voxel_sizes`. Reading it raised AttributeError, and because the
prediction-visualisation window is built before that line runs, the only symptom
was an empty window. A syntax check cannot see this, so scan for it.
"""
import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

MODULES = [
    "src/ml_threshold_selection/ui_visualization.py",
    "src/ml_threshold_selection/analysis_pipeline.py",
    "src/ml_threshold_selection/data_io.py",
    "src/ml_threshold_selection/fabric_pipeline.py",
]

# Attributes created after construction, on the paths that set them.
ASSIGNED_LATER = {
    "test_voxel_size_mm", "visualization_window", "ellipsoid_analysis_results",
    "test_file_path", "training_files", "probabilities", "features",
    "training_results", "resolution_aware_engineer", "root", "model",
    "test_data", "training_data", "expert_thresholds", "sample_list",
    "test_voxel_sizes", "training_voxel_sizes", "ellipsoid_feature_analyzer",
    "log", "status_text", "threshold_config",
}


def _app_attributes():
    """Attribute names the application object actually carries.

    Collected from every module that writes to it, since the helper modules
    assign onto `app` as well as the controller assigning onto `self`. Parsed
    rather than imported: importing the controller pulls in the whole plotting
    stack, which a test about attribute names does not need.
    """
    names = set()
    package = ROOT / "src" / "ml_threshold_selection"
    for path in sorted(package.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id in ("self", "app")
                    and isinstance(node.ctx, ast.Store)):
                names.add(node.attr)
            elif isinstance(node, ast.FunctionDef):
                names.add(node.name)
    return names


def _attributes_read_from_app(path):
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    found = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "app"
                and isinstance(node.ctx, ast.Load)):
            found.add(node.attr)
    return found


@pytest.mark.parametrize("relative", MODULES)
def test_module_only_reads_attributes_the_app_has(relative):
    path = ROOT / relative
    if not path.is_file():
        pytest.skip(f"{relative} is not present")
    available = _app_attributes() | ASSIGNED_LATER
    unknown = sorted(a for a in _attributes_read_from_app(path)
                     if a not in available and not a.startswith("_"))
    assert not unknown, (
        f"{relative} reads app attributes that are never assigned: {unknown}. "
        "Reading one raises AttributeError at run time."
    )


def test_the_voxel_size_attribute_is_named_consistently():
    """Guard the specific name that was wrong."""
    for relative in MODULES:
        path = ROOT / relative
        if not path.is_file():
            continue
        assert "app.voxel_sizes" not in path.read_text(encoding="utf-8"), (
            f"{relative} uses app.voxel_sizes; the attribute is "
            "app.training_voxel_sizes"
        )
