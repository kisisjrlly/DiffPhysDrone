Use case: scientific-educational
Task type: edit
Asset type: updated SCI journal method figure for "Differentiable Active-Depth Model"

Primary request:
Edit the provided v19 figure to match the revised paper wording and code interface. Preserve the left input stack, active-stereo wall sketch, S_phi factor rows, pictorial thumbnails, and overall polished style. Update only the right-side sensor-output logic and labels.

Revised paper/code semantics to show:
- The sensor model S_phi publicly returns only two outputs:
  (D_obs, Q_obs)
- Do not show M, M_t, M_ST, valid_prob, or "soft validity map" as public or named outputs.
- Internally, the quality response Q determines a binary quality gate V = 1[Q > q0].
- Observed depth and observed quality are:
  D_obs = D_raw x V
  Q_obs = Q x V
- Teacher fill is a downstream depth-health operator, not a direct sensor output. Do not add teacher/fill/loss boxes in this figure.

Right-side layout correction:
- Keep the "quality Q" box as an internal quality response produced by the five S_phi factor rows.
- Replace the two boxes currently labeled "soft validity M" and "straight-through mask M_ST" with one compact internal gate box:
  "quality gate V"
  "V = 1[Q > q0]"
- Draw Q -> quality gate V.
- Draw V feeding both output compositions.
- Draw D_raw (fixed depth) feeding D_obs.
- Draw Q feeding Q_obs.
- Draw only two output boxes on the far right:
  "observed depth D_obs" with formula "D_obs = D_raw x V"
  "observed quality Q_obs" with formula "Q_obs = Q x V"
- Add a small label near the far-right outputs:
  "S_phi outputs: (D_obs, Q_obs)"

Left and center must remain:
- input stack with native renderer / raw depth D_raw, scene/material mask Omega, speed ||v_t||, camera registers c=[P,E,G]
- row dependencies:
  active return: (P, D_raw)
  ambient washout: (E, Omega)
  motion blur: (E, ||v_t||)
  gain/noise: (G)
  edge difficulty: (D_raw)
- all five rows merge into quality Q.

Strict text removals:
- Remove all occurrences of "soft validity M".
- Remove all occurrences of "M_ST".
- Remove "straight-through mask".
- Do not write valid_prob.
- Do not write M as a standalone variable.

Strict exclusions:
- No red arrows.
- No gradient/backward path.
- No loss box.
- No teacher fill box.
- No dense formulas.
- No major redesign of the left or center visual content.

Style:
Keep a clean, high-quality, vector-like SCI journal diagram with readable labels and clear black/dark-gray forward arrows.
