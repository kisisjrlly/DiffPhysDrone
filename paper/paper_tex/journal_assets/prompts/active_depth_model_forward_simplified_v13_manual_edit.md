Use case: scientific-educational
Task type: edit
Asset type: SCI journal method figure for "Differentiable Active-Depth Model"

Primary request:
Edit the provided v12 figure with small, precise corrections to the forward data-flow arrows and labels. Keep the visual style, polished pictorial thumbnails, wall/aperture sketch, wide layout, teal/orange frames, and overall composition. Do not redesign from scratch.

Scientific implementation to match:
In env_cuda.py, render_diff_depth first calls native render_depth to produce raw geometric depth D_raw. Then the sensor branch computes a scene/material mask Omega from raw depth, takes speed ||v_t|| as detached context, and takes camera registers c=[P,E,G] as the controllable sensor input. The active_sensing_sensor produces quality Q, soft validity M, hard/straight-through validity M_ST, observed depth D_obs, and observed quality Q_obs.

Correct forward data flow to draw:
1. Native renderer / active stereo sketch produces raw depth D_raw.
2. D_raw and Omega are fixed inputs into S_phi, not inputs into camera registers.
3. Camera registers c=[P,E,G] are an independent input into S_phi, not an output of the renderer or geometry stream.
4. Speed ||v_t|| is a small context input into the motion blur module only.
5. Inside S_phi:
   - active return uses (P, D_raw)
   - ambient washout uses (E, Omega)
   - motion blur uses (E, ||v_t||)
   - gain/noise uses (G)
   - edge difficulty uses (D_raw)
6. All five factor modules visually merge into quality Q. Do not connect only active return to Q.
7. Then draw Q -> soft validity M -> straight-through mask M_ST.
8. Draw output composition explicitly:
   - D_obs = D_raw x M_ST
   - Q_obs = Q x M_ST
   D_raw must visibly participate in D_obs. Q must visibly participate in Q_obs. M_ST must feed both outputs.

Important arrow corrections:
- Remove any arrow from the active stereo sketch or geometry stream into "Camera registers c=[P,E,G]".
- Remove any arrow from D_raw or Omega into the camera-register box.
- Add clean black arrows from D_raw and Omega directly into the S_phi sensor block / relevant factor modules.
- Add a clean black arrow from camera registers c=[P,E,G] into the S_phi sensor block / relevant factor modules.
- Add a clean black arrow from speed ||v_t|| into the motion blur row only.
- Add a clear combiner or bracket showing all five rows contribute to quality Q.
- Use only black/dark-gray forward arrows. Do not draw red dashed arrows, gradient arrows, or a loss box.

Label corrections:
- Change the bottom-left note to: "Omega in [0,1]: scene/material mask".
- If space allows, write soft validity as "M = sigmoid((Q - q0) / sigma_q)".
- Keep compact formulas only:
  "Q = clip(Q0 + Omega C_rho, 0, 1)"
  "M = sigmoid((Q - q0) / sigma_q)"
  "D_obs = D_raw x M_ST"
  "Q_obs = Q x M_ST"

Visual constraints:
- Preserve the high-quality pictorial mini-panels from v12: wall/aperture sketch, range/return image, washed/noisy surface, motion blur, speckled gain noise, edge difficulty heatmap, depth and quality maps.
- Keep the figure readable and less formula-heavy than the original source image.
- Keep all labels large and horizontal.
- Maintain a clean SCI journal vector-like style.
- Do not imply S_phi outputs x, c, or camera state.
- Do not imply the native renderer R is differentiable inside the sensor branch.

Avoid:
- No red lines.
- No gradient text.
- No loss box.
- No generic over-simplified icons replacing the current phenomenon thumbnails.
- No arrows that enter the camera-register box from D_raw, Omega, or the active stereo sketch.
