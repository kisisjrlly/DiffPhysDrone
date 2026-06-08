Use case: scientific-educational
Task type: edit
Asset type: final simplified SCI journal method figure for "Differentiable Active-Depth Model"

Primary request:
Polish the provided simplified active-depth model figure. Keep the current clean layout, labels, thumbnails, colors, and module arrangement. Make only small visual corrections so the data flow is clearer and publication-ready.

Must preserve:
- White background, teal headers, pale cyan sensor blocks, black forward arrows, red backward gradient arrows.
- Left-to-right flow: Inputs/native geometry -> S_phi differentiable active-depth sensor -> soft validity M -> straight-through mask -> D_obs and Q_obs -> L_sens / L_teach.
- The left input stack: Native renderer R -> D_raw, scene mask Omega, speed ||v_t||, camera registers c=[P,E,G] with P/E/G chips.
- The central S_phi block and the five factor chips:
  "active return (P, range)"
  "ambient washout (E, Omega)"
  "motion blur (E, ||v_t||)"
  "gain/noise (G)"
  "edge difficulty (depth discontinuity)"
- The gray note: "D_raw, Omega, speed, and R are fixed in sensor backward".

Required correction:
- Make the black forward arrow from "camera registers c=[P,E,G]" into the S_phi block clean and readable.
- The black camera-register input arrow must not visually merge with the red gradient arrow.
- The red gradient should clearly end at the camera registers / P,E,G chips only.
- Keep the red label "gradient to P,E,G" close to the red path.
- Do not draw any red arrow to D_raw, scene mask Omega, speed ||v_t||, or Native renderer R.

Accuracy constraints:
- S_phi outputs observed depth and observed quality through Q, M, and the straight-through mask.
- S_phi does not output state x or camera state c.
- D_raw, Omega, speed, and R are fixed in the sensor backward path.
- Gradients from L_sens / L_teach go back to P, E, and G only.

Avoid:
- Do not add formulas.
- Do not add extra panels or regime examples.
- Do not change variable names.
- Do not shrink text.
- Do not introduce dense wiring or crossing arrows.
- Do not make the figure look hand-drawn.
