Use case: scientific-educational
Task type: edit
Asset type: SCI journal method figure for "Differentiable Active-Depth Model"

Primary request:
Edit the provided active-depth model figure into a cleaner, less formula-heavy, publication-quality diagram. Use the provided image as the direct edit target and preserve its strong pictorial style, especially the small phenomenon thumbnails inside the modules. The goal is not to make an abstract icon diagram; the goal is to make the original scientific mechanism easier to read.

Core scientific message:
A native geometric renderer produces raw depth D_raw and a scene mask Omega. Camera registers c=[P,E,G] control a differentiable active-depth sensor S_phi. S_phi models active return, ambient washout, motion blur, gain/noise, and depth-edge difficulty. These factors produce a quality map Q and soft validity M. A straight-through mask gives observed depth D_obs and observed quality Q_obs.

Required layout:
- Keep the same overall left-to-right structure and wide aspect ratio as the input image.
- Left: active stereo sensing sketch with a real-looking grid/wall target and red projector rays. This sketch should remain visually polished and recognizable as a wall / aperture scene.
- Left lower: "Geometry stream" with two map panels:
  "raw depth D_raw"
  "scene mask Omega"
- Middle-left: "Camera registers c=[P,E,G]" with three vertical controls:
  "P projector power"
  "E exposure"
  "G gain"
- Center: big teal-framed block titled "S_phi differentiable active-depth sensor".
- Inside S_phi, keep five horizontal or stacked factor modules with clear pictorial thumbnails, not generic empty icons:
  1. "active return" with a range attenuation curve or laser return visual; label "(P, D_raw)"
  2. "ambient washout" with a bright noisy/washed surface example; label "(E, Omega)"
  3. "motion blur" with blurred streaks or a motion example; label "(E, ||v_t||)"
  4. "gain / noise" with speckled sensor noise texture; label "(G)"
  5. "edge difficulty" with a depth discontinuity / edge patch; label "(D_raw)"
- Right of S_phi: compact boxes for:
  "quality Q"
  "soft validity M"
  "straight-through mask"
  "D_obs observed depth"
  "Q_obs observed quality"

Formula reduction:
- Remove most equations and dense formulas from the original.
- Keep only variable names and a few compact relations that are essential:
  "Q = clip(Q0 + Omega C_rho, 0, 1)"
  "M = sigmoid(Q)"
  "D_obs = D_raw * M_ST"
  "Q_obs = Q * M_ST"
- Do not include long SNR, active signal, passive signal, noise, washout, or edge equations.

Forward data-flow arrows:
- Use only black or dark gray arrows.
- D_raw and Omega feed into S_phi.
- c=[P,E,G] feeds into S_phi and its factor modules.
- speed ||v_t|| may be shown as a small fixed context input feeding motion blur only.
- The five factor modules merge into quality Q.
- Q flows to M, M flows to straight-through mask, and the mask produces D_obs and Q_obs.

Important constraints:
- Do NOT draw any red dashed gradient/backward lines.
- Do NOT draw any loss box.
- Do NOT imply S_phi outputs x or camera state c.
- Do NOT imply the native renderer R is differentiable in the sensor branch.
- Keep all text large enough to read in a paper.
- Keep the high-quality pictorial mini-panels from the original style; make them cleaner if needed, not simpler.
- Maintain the polished vector-like SCI journal style: white background, teal module frames, orange/red quality/validity boxes, consistent alignment, clean arrows.

Avoid:
- Avoid over-simplified generic icons.
- Avoid replacing the wall/aperture sketch with an unrelated object.
- Avoid tiny formulas, dense explanatory prose, watermarks, decorative clutter, and misspelled variables.
