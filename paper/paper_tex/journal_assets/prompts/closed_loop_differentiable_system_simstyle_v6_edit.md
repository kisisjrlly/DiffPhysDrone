Use case: scientific-educational
Task type: edit
Asset type: corrected SCI journal time-unrolled differentiable-system figure, 1536x512

Edit target invariants:
- Preserve the current time-unrolled differentiable-simulation style: pale blue panels, state circles, small grayscale depth thumbnails, compact policy blocks, black forward arrows, red backward-gradient arrows, and right-side legend.
- Preserve the repeated time-step layout: k=0, k=1, k, k=T-1.
- Preserve the core modules: x_k,c_k, S_phi, D_obs,k, Pi_theta, Pi_f, Pi_c, u_k, c*_{k+1}, EMA, f_dyn, L_sens/L_teach, L_flight.
- Do not add Q_t or M_t. Do not add a separate R renderer.

Primary correction 1: forward camera update
- In every time-step panel, make the camera update explicit:
  c_k and c*_{k+1} both feed into EMA.
  EMA outputs c_{k+1}.
- Draw a clear black arrow from the state/camera path "c_k" into EMA.
- Draw a clear black arrow from "Pi_c -> c*_{k+1}" into EMA.
- Label the EMA output as "c_{k+1}".

Primary correction 2: forward UAV dynamics update
- In every time-step panel, make the dynamics update explicit:
  u_k and x_k feed into f_dyn.
  f_dyn outputs x_{k+1}.
- Draw a clear black arrow from "Pi_f -> u_k" into f_dyn.
- Draw a clear black arrow from the state path "x_k" into f_dyn.
- Label the f_dyn output as "x_{k+1}".

Primary correction 3: next-state merge
- The next state circle must visibly combine two separate outputs:
  x_{k+1} from f_dyn and c_{k+1} from EMA.
- Draw two clean black arrows that merge into the next state circle labeled "x_{k+1}, c_{k+1}".
- Avoid ambiguous bus lines or arrows that make EMA look like it feeds f_dyn.

Primary correction 4: backward gradients
- Keep the black forward graph unchanged except for the corrections above.
- Red gradient from "L_flight" should go back through f_dyn, u_k, Pi_f/Pi_theta, D_obs,k, and S_phi.
- Do NOT draw a solid red gradient from L_flight into Pi_c or EMA, because camera branch is frozen/detached during flight-only adaptation.
- Red gradient from "L_sens / L_teach" should go back to S_phi and to camera register c_k. Label this camera-register gradient "dL/dc_k".
- If a camera-side gradient availability must be indicated, use a faint dashed red arrow labeled "available / frozen" near Pi_c, not a solid update arrow.

Primary correction 5: perception-to-policy path
- Keep S_phi -> D_obs,k -> Pi_theta visually dominant and clear.
- D_obs,k must be represented as the grayscale depth thumbnail.
- A black arrow from the depth thumbnail must enter Pi_theta.

Policy relation:
- Pi_theta is one split policy with two heads:
  Pi_f -> u_k
  Pi_c -> c*_{k+1}
- Pi_f and Pi_c are parallel heads, not sequential policies.
- Do not draw arrows from Pi_f to Pi_c or from Pi_c to Pi_f.

Style:
- Keep labels large and readable.
- Use smooth clean arrows with minimal crossings.
- Keep the figure elegant and close to the provided differentiable-simulation reference style.

Avoid:
Tiny text, long equations, Q_t, M_t, separate R renderer, EMA feeding f_dyn, f_dyn feeding EMA, L_flight solid red arrows into Pi_c/EMA, missing c_k -> EMA, missing x_k -> f_dyn, ambiguous next-state merge, watermark, logo.
