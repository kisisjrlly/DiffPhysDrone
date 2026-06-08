Use case: scientific-educational
Task type: edit
Asset type: simplified SCI journal method figure for "Differentiable Active-Depth Model"
Primary request: Redesign the provided complex active-depth model figure into a clean, simplified, publication-quality diagram. Remove most formulas and dense text. Show only the essential differentiable sensing flow and correct gradient destinations.

Input images:
- Image 1: original complex active-depth model figure, used as reference for style and visual metaphors.

Core message:
Raw geometric depth and scene mask are fixed inputs to a differentiable active-depth sensor. Camera registers P/E/G control sensor quality and validity. The sensor outputs observed depth and quality maps. Loss gradients flow back only to P/E/G, not to raw geometry, scene mask, speed, or renderer.

Required layout, left to right:
1. Inputs / native geometry:
   - small active stereo sketch
   - "Native renderer R" -> "D_raw"
   - "scene mask Omega"
   - "speed ||v_t||"
   - "camera registers c=[P,E,G]" with three visible chips: "P power", "E exposure", "G gain"
2. Central block:
   - big box "S_phi Differentiable active-depth sensor"
   - five small factor chips only:
     "active return (P, range)"
     "ambient washout (E, Omega)"
     "motion blur (E, ||v_t||)"
     "gain/noise (G)"
     "edge difficulty (depth discontinuity)"
   - these merge into "quality Q"
3. Validity and mask:
   - "soft validity M"
   - "straight-through mask: hard forward, soft backward"
4. Outputs:
   - "D_obs observed depth" with grayscale thumbnail
   - "Q_obs observed quality" with heatmap thumbnail
   - small loss box "L_sens / L_teach"

Forward arrows:
- Use black arrows only.
- D_raw, Omega, speed, and c=[P,E,G] feed into S_phi / factor chips.
- Q -> M -> D_obs and Q_obs.

Backward arrows:
- Use red arrows only.
- Draw a SHORT, clean red gradient path from "L_sens / L_teach" through Q/M/D_obs back to "camera registers c=[P,E,G]".
- Label the red path "gradient to P,E,G".
- Absolutely do NOT draw red arrows to speed ||v_t||.
- Absolutely do NOT draw red arrows to Omega, D_raw, active stereo sketch, or renderer R.
- Add a gray note: "D_raw, Omega, speed, and R are fixed in sensor backward".

Important visual constraints:
- No long red bus along the bottom of the figure.
- No red arrowheads under the speed box.
- No red arrow to scene/material mask.
- Keep the figure less crowded than the original.
- Make labels large enough for a paper figure.
- Preserve a polished vector-like SCI journal style.

Avoid:
- Avoid dense equations and long formulas.
- Avoid many tiny thumbnails or regime panels.
- Avoid blacking out text or replacing P/E/G chips with blank boxes.
- Avoid implying geometry-ray intersections are differentiable in this sensor branch.
