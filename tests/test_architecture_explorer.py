# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Focused checks for the YOLO Architecture Explorer training graph."""

import importlib.util
import sys
import subprocess
import shutil
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
EXPLORER = ROOT / "examples/YOLO-Architecture-Explorer"


def _load_builder():
    """Load the explorer builder without making the example a package."""
    sys.path.insert(0, str(EXPLORER))
    spec = importlib.util.spec_from_file_location("architecture_explorer_builder", EXPLORER / "build_explorer.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("subdir", "config", "scale", "levels", "expected_losses", "absent_losses", "assigner"),
    [
        ("v8", "yolov8.yaml", "n", ["P3", "P4", "P5"], {"CIoU", "DFL", "BCE"}, set(), "TAL"),
        ("v8", "yolov8-p2.yaml", "n", ["P2", "P3", "P4", "P5"], {"CIoU", "DFL", "BCE"}, set(), "TAL"),
        ("26", "yolo26.yaml", "n", ["P3", "P4", "P5"], {"CIoU", "BCE"}, {"DFL"}, "TAL"),
        ("rt-detr", "rtdetr-l.yaml", "l", ["P3", "P4", "P5"], {"L1", "GIoU", "VFL"}, {"CIoU", "DFL"}, "HungarianMatcher"),
    ],
)
def test_learning_graph_is_derived_from_real_head(
    subdir, config, scale, levels, expected_losses, absent_losses, assigner
):
    """The learning graph follows the built head and criterion instead of a fixed HTML-only mapping."""
    builder = _load_builder()
    model, _ = builder.build_model(ROOT / "ultralytics/cfg/models" / subdir / config, scale)
    graph = builder.learning_loop(model)

    assert graph["head_graph"]["name"] == type(model.model[-1]).__name__
    assert [x["name"] for x in graph["head_graph"]["levels"]] == levels
    loss_names = {x["name"] for x in graph["losses"]}
    assert loss_names >= expected_losses
    assert loss_names.isdisjoint(absent_losses)
    assert graph["head_graph"]["assigner"].startswith(assigner)


def test_template_renders_learning_graph_inside_final_head_card():
    """The interactive card path must attach learning/loss nodes onto the real final head structure in train mode."""
    source = (EXPLORER / "template.html").read_text()
    assert "function attachLearningNodes" in source
    assert "const isHeadChain" in source
    assert 'S.phase === "train"' in source
    assert "attachLearningNodes(lay, cfg, `c${ci}_`)" in source
    assert "l?.losses" in source


def test_template_javascript_parses():
    """A quote or template-literal error must not blank the entire explorer."""
    source = (EXPLORER / "template.html").read_text()
    node = shutil.which("node")
    if not node:
        pytest.skip("node is required to parse the browser bundle")
    script = """
const fs = require('fs'), vm = require('vm');
const html = fs.readFileSync(0, 'utf8');
const code = [...html.matchAll(/<script(?: [^>]*)?>([\\s\\S]*?)<\\/script>/g)]
  .map((match) => match[1]).find((value) => value.includes('const DATA'));
new vm.Script(code);
"""
    subprocess.run([node, "-e", script], input=source, text=True, check=True)
