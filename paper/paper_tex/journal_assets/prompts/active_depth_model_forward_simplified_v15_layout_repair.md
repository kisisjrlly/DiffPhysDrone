Use case: scientific-educational
Task type: edit
Asset type: corrected SCI journal method figure for "Differentiable Active-Depth Model"

Primary request:
Repair the layout of the provided figure so the forward data flow is unambiguous and matches the paper/code. Keep the polished pictorial scientific style, white background, teal/orange frames, wall/aperture sketch, and phenomenon thumbnails, but it is acceptable to reposition the left input boxes to make arrows correct.

Make the left side an input stack, not a blocked routing area:
- Put these as four independent input/context items on the left:
  1. Native renderer R -> raw depth D_raw
  2. scene/material mask Omega
  3. speed ||v_t||
  4. camera registers c=[P,E,G] with P, E, G controls
- Each item should have its own clean black arrow into S_phi or the relevant row.
- Camera registers c=[P,E,G] must NOT receive any arrow. It only sends arrows to S_phi.
- D_raw and Omega must NOT enter the camera-register box.

Correct S_phi row dependencies:
- active return row receives P and D_raw; label "(P, D_raw)".
- ambient washout row receives E and Omega; label "(E, Omega)".
- motion blur row receives E and speed ||v_t||; label "(E, ||v_t||)".
- gain/noise row receives G; label "(G)".
- edge difficulty row receives D_raw; label "(D_raw)".
- Draw a clear bracket/common combiner so all five rows jointly produce quality Q.

Correct right-side output chain:
- Draw Q -> soft validity M -> straight-through mask M_ST.
- Then draw M_ST as a mask/gate feeding both observed outputs.
- Draw D_raw as a thin fixed-depth input to the D_obs composition only.
- Draw Q as a thin quality input to the Q_obs composition only.
- Avoid any line that looks like Q directly creates D_obs.
- Keep compact equations:
  "Q = clip(Q0 + Omega C_rho, 0, 1)"
  "M = sigmoid((Q - q0) / sigma_q)"
  "D_obs = D_raw x M_ST"
  "Q_obs = Q x M_ST"

Visual content to preserve:
- active stereo sensing with IR projector and wall/aperture target
- raw depth grayscale thumbnail
- scene mask thumbnail
- active return visual / range attenuation visual
- ambient washout noisy/washed visual
- motion blur visual
- gain/noise speckle visual
- edge difficulty/depth discontinuity visual
- quality, validity, observed depth, and observed quality map thumbnails

Strict exclusions:
- No red arrows.
- No gradient/backward path.
- No loss box.
- No arrows into camera registers.
- No dense formulas or paragraph text.
- No generic icons replacing phenomenon thumbnails.

Style:
Top-tier SCI/CS journal method figure, clean vector-like diagram with high-quality pictorial subpanels, large readable labels, aligned modules, and clear arrowheads.
