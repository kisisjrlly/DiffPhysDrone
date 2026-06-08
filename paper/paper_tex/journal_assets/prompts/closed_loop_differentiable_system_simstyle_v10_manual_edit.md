Use case: scientific-educational
Task type: edit
Asset type: final wide SCI journal method figure, 1536x512 PNG
Primary request: Make a small correction to the provided v9 diagram while preserving its polished journal style and layout. The current v9 is visually good; only refine the red backward-gradient semantics around L_flight and the camera branch.

Input images:
- Image 1: current v9 diagram. Preserve its overall composition, typography, white background, clean arrows, module layout, and depth-observation image.

Edit target invariants:
- Keep the same polished vector-like journal style.
- Keep one single Pi_theta policy box. Do NOT split into Pi_f/Pi_c.
- Keep S_phi feeding D_obs,t into Pi_theta.
- Keep Q_t, M_t as a side output for L_sens/L_teach only; do NOT feed Q_t, M_t into Pi_theta.
- Keep x_t,c_t as inputs to S_phi; S_phi must not output x_t or c_t.
- Keep Pi_theta outputting both u_t and c*_{t+1}.
- Keep u_t -> f_dyn -> x_{t+1}.
- Keep c*_{t+1} and c_t -> EMA -> c_{t+1}.

Required correction:
- The red gradient from L_flight must clearly go to u_t / f_dyn / x_{t+1}, and then backward through Pi_theta and D_obs,t to S_phi.
- Remove or convert to dashed any solid red arrow that appears to update the camera output c*_{t+1}, EMA, or camera branch during flight-only.
- Near c*_{t+1}/EMA, use only a dashed red stopped/optional indicator with the label "flight-only: sg[c*], theta_cam fixed".
- L_flight should not have a solid red arrow into EMA or c*_{t+1}. It may have a dashed optional/stopped arrow near that branch.
- If possible, add a small red label near the main L_flight path: "via u_t, x_{t+1}".

Keep everything else the same. Do not redesign the figure. Do not add clutter. Do not move major boxes unless necessary for readability.

Avoid:
- Avoid new variables.
- Avoid extra policy boxes.
- Avoid ambiguous red arrows into the frozen camera branch.
- Avoid making the figure uglier or more crowded.
