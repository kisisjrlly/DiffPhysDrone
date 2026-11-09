"""The interaction probe's decision rule must be a statistical test, not argmax.

History: the first rule took ``argmax`` of accuracy over the exposure grid per
speed and regressed peak exposure on speed. On a reduced-light run it returned
``R1_INTERACTION_EXISTS`` with a slope of -11.6 ms/(m/s) - and that verdict was
wrong. The cells it depended on were within two or three test samples of each
other, accuracy and log-loss disagreed in *sign* at the driving cell, and
argmax is discontinuous so one sample could swing the slope by ~11.

These tests pin the replacement: a 2x2 difference-of-differences contrast on the
long-exposure penalty, with a standard error from all four cells, and a
requirement that the two readouts agree in sign.
"""

import importlib.util
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

_spec = importlib.util.spec_from_file_location(
    "exposure_probe",
    os.path.join(ROOT, "tools", "audit_exposure_motion_probe.py"),
)
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)

SHORT = 0.89
LONG = 8.0


def _cell(speed, exposure, accuracy, log_loss, n=400):
    return {
        "speed_m_s": speed,
        "exposure_ms": exposure,
        "accuracy": accuracy,
        "n_test": n,
        "accuracy_se": math.sqrt(max(accuracy * (1.0 - accuracy), 0.0) / n),
        "log_loss": log_loss,
    }


def _grid(penalties, *, base_short=0.90, log_penalty_scale=0.30, n=400):
    """Build a grid from a per-speed list of (speed, accuracy_penalty)."""
    cells = []
    for speed, penalty in penalties:
        short_acc = base_short
        long_acc = base_short - penalty
        cells.append(_cell(speed, SHORT, short_acc, 0.40, n))
        cells.append(_cell(speed, LONG, long_acc, 0.40 + penalty * log_penalty_scale, n))
    return cells


def test_detects_a_genuine_interaction():
    """Penalty absent at low speed, large at high speed: this is coupling."""
    cells = _grid([(0.3, 0.00), (0.6, 0.10), (1.15, 0.25)], n=400)
    result = probe.interaction_test(cells)

    assert result["detected"] is True
    assert result["interaction_z"] >= 2.0
    assert result["interaction_contrast"] > 0.2
    assert result["readouts_agree_on_interaction"] is True


def test_saturated_baseline_with_equal_penalty_is_not_an_interaction():
    """A penalty that exists at every speed is saturation, not coupling.

    This is the case the first rule got wrong: it required the penalty to be
    *absent* at low speed, which can never happen in a saturating camera, so a
    perfectly flat penalty was at risk of being read either way. The contrast
    must be ~0 and the verdict must not be R1.
    """
    cells = _grid([(0.3, 0.15), (0.6, 0.15), (1.15, 0.15)], n=400)
    result = probe.interaction_test(cells)

    assert abs(result["interaction_contrast"]) < 1e-9
    assert result["detected"] is False
    assert result["underpowered"] is True
    # Both speeds are individually penalised - which is exactly why "penalty
    # present" cannot be used as evidence of interaction.
    assert result["slow_speed_penalised"] is True
    assert result["fast_speed_penalised"] is True

    classification = probe.classify(cells, slope_threshold=1.0, monotone_margin=0.01,
                                    interaction=result)
    assert classification["verdict"] != "R1_INTERACTION_EXISTS"
    assert classification["blocks_training"] is True


def test_readout_disagreement_blocks_detection():
    """Accurate contrast in accuracy but the opposite sign in log-loss.

    A criterion that can be flipped by which readout you print is not a
    criterion, so this must not be reported as interaction even though the
    accuracy z is far past threshold.
    """
    cells = []
    for speed, penalty, log_penalty in ((0.3, 0.00, 0.00), (1.15, 0.25, -0.05)):
        cells.append(_cell(speed, SHORT, 0.90, 0.40))
        cells.append(_cell(speed, LONG, 0.90 - penalty, 0.40 + log_penalty))

    result = probe.interaction_test(cells)

    assert result["interaction_z"] >= 2.0          # accuracy says yes
    assert result["log_loss_interaction_contrast"] < 0.0   # log-loss says no
    assert result["readouts_agree_on_interaction"] is False
    assert result["detected"] is False


def test_small_sample_is_reported_as_underpowered_not_as_no_effect():
    """With few samples a real effect must be flagged unresolvable, not denied.

    Conflating "we cannot see it" with "it is not there" is what let the earlier
    rule's null result look like a finding.
    """
    cells = _grid([(0.3, 0.00), (1.15, 0.06)], n=24)
    result = probe.interaction_test(cells)

    assert result["underpowered"] is True
    assert result["detected"] is False
    classification = probe.classify(cells, slope_threshold=1.0, monotone_margin=0.01,
                                    interaction=result)
    assert classification["verdict"] == "R5_UNDERPOWERED"
    assert classification["blocks_training"] is True


def test_only_r1_permits_training():
    """Every other verdict must block; this is the property require_gates relies on."""
    interacting = probe.interaction_test(_grid([(0.3, 0.0), (1.15, 0.3)], n=400))
    flat = probe.interaction_test(_grid([(0.3, 0.15), (1.15, 0.15)], n=400))

    allow = probe.classify(_grid([(0.3, 0.0), (1.15, 0.3)], n=400),
                           slope_threshold=1.0, monotone_margin=0.01,
                           interaction=interacting)
    block = probe.classify(_grid([(0.3, 0.15), (1.15, 0.15)], n=400),
                           slope_threshold=1.0, monotone_margin=0.01,
                           interaction=flat)

    assert allow["verdict"] == "R1_INTERACTION_EXISTS"
    assert allow["blocks_training"] is False
    assert block["blocks_training"] is True


def test_accuracy_se_shrinks_with_sample_size():
    """The standard error must actually use the sample count."""
    few = _cell(1.15, SHORT, 0.9, 0.4, n=100)["accuracy_se"]
    many = _cell(1.15, SHORT, 0.9, 0.4, n=1600)["accuracy_se"]
    assert few == many * 4.0


def test_readout_ceiling_overrides_an_apparent_interaction():
    """A saturated readout must not be allowed to report coupling.

    Observed on the single-wall geometry: four exposures per speed read ~1.000,
    and the reported R1 (z=+4.97) came entirely from one cell reading 0.844 at
    0.89 ms while 0.4 ms read 1.000 at the same speed. When the classifier
    already separates the classes perfectly there is no dynamic range left for
    exposure to act on, so the remaining variation is noise.
    """
    cells = []
    for speed, short_acc, long_acc in ((0.3, 0.844, 1.000),
                                       (0.6, 1.000, 0.992),
                                       (1.15, 1.000, 0.992)):
        cells.append(_cell(speed, SHORT, short_acc, 0.45, n=128))
        cells.append(_cell(speed, LONG, long_acc, 0.45, n=128))
    # Fill the grid with near-ceiling cells, as the real run had.
    for speed in (0.3, 0.6, 1.15):
        for exposure in (0.2, 0.4, 20.0):
            cells.append(_cell(speed, exposure, 1.0, 0.44, n=128))

    interaction = probe.interaction_test(cells)
    ceiling = probe.readout_ceiling(cells)
    classification = probe.classify(cells, slope_threshold=1.0, monotone_margin=0.01,
                                    interaction=interaction, ceiling=ceiling)

    assert ceiling["degenerate"] is True
    assert ceiling["cells_at_ceiling"] >= 4
    assert classification["verdict"] == "R6_READOUT_CEILING"
    assert classification["blocks_training"] is True
    assert classification["interaction_present"] is False
    # The message must say the readout cannot answer, not that coupling is absent.
    assert "cannot answer" in classification["meaning"]


def test_ceiling_guard_leaves_a_hard_grid_alone():
    """A grid with real dynamic range must not be dismissed as degenerate."""
    cells = []
    for speed in (0.3, 1.15):
        for exposure, acc in ((SHORT, 0.95), (LONG, 0.65)):
            cells.append(_cell(speed, exposure, acc, 0.5, n=256))
        for exposure in (0.2, 0.4, 20.0):
            cells.append(_cell(speed, exposure, 0.8, 0.5, n=256))

    ceiling = probe.readout_ceiling(cells)
    assert ceiling["degenerate"] is False
