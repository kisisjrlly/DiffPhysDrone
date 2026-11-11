"""Scenes with differentiable pose rendering and continuous body collision."""

DIFFERENTIABLE_SCENES = (
    "intent_gates",
    "sparse_poles",
    "joint_slit",
    "occluded_route",
    # Exact single-wall geometry used by the historical grayscale checkpoint,
    # rendered through the differentiable pose/exposure path.  This is an
    # explicit opt-in alias; the legacy ``single_wall_slit`` behavior remains
    # unchanged for regression comparisons.
    "single_wall_slit_diff",
)
