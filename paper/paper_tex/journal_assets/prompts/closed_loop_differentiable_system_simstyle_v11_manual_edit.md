Use case: scientific-educational
Task type: edit
Asset type: final wide SCI journal method figure, 1536x512 PNG
Primary request: Edit the provided v9 diagram into a final polished journal figure. Preserve v9's clean style and layout, but fix the L_flight backward-gradient arrows so they cannot be misread as updating the camera branch.

Input images:
- Image 1: v9 diagram. Use it as the base. It is visually good.

VERY IMPORTANT: Preserve all black forward arrows from v9 unless directly contradicted below. Preserve the module positions, white background, depth image, typography, and colors.

Required red gradient layout:
1. L_flight should have solid red gradient arrows ONLY along the flight/task path:
   L_flight -> x_{t+1} / f_dyn,
   L_flight -> u_t / Pi_theta,
   Pi_theta -> D_obs,t,
   D_obs,t -> S_phi.
2. Add a small red label near this path: "via u_t, x_{t+1}".
3. Remove all solid red arrows from L_flight to c*_{t+1}, EMA, c_{t+1}, psi_cam, or the camera update branch.
4. Near c*_{t+1} and EMA, keep only a dashed red stopped/optional path labeled exactly: "flight-only: sg[c*], theta_cam fixed".
5. There must be NO solid red arrow pointing into psi_cam.
6. There must be NO solid red arrow pointing into EMA.
7. There must be NO solid red arrow from L_flight into c*_{t+1}.

Other required data-flow correctness:
- One single Pi_theta policy box only, with outputs "u_t, c*_{t+1}".
- Do not create separate Pi_f or Pi_c boxes.
- S_phi outputs D_obs,t to Pi_theta.
- S_phi has a side output Q_t, M_t to L_sens/L_teach.
- Q_t, M_t do not feed into Pi_theta.
- S_phi does not output x_t or c_t.
- c_t and c*_{t+1} both feed EMA; EMA outputs c_{t+1}.
- u_t feeds f_dyn; f_dyn outputs x_{t+1}.

Keep L_sens/L_teach arrows:
- L_sens/L_teach should point through Q_t, M_t and S_phi back toward camera parameters / c_t.
- This part can remain similar to v9.

Avoid:
- Do not move boxes around unnecessarily.
- Do not introduce new red arrows into the camera encoder or camera update branch.
- Do not add clutter.
- Do not make the diagram rougher; keep it journal-ready.
