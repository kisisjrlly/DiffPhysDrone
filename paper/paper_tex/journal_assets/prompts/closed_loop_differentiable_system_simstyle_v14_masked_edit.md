Use case: scientific-educational
Task type: precise-object-edit
Asset type: final wide SCI journal method figure, 1536x512 PNG
Primary request: Make a tiny masked edit to the provided v9 scientific diagram. Preserve everything outside the mask exactly. Inside the mask, remove the ambiguous solid red backward-gradient arrow from L_flight to the camera output c*_{t+1}. Do not change the surrounding layout.

Edit target invariants:
- Outside the mask, keep the image unchanged.
- Keep the black forward arrow from Pi_theta output c*_{t+1} to EMA.
- Keep the Pi_theta box and its text unchanged.
- Keep the L_flight box unchanged.
- Do not add any red arrow to c*_{t+1}, EMA, or psi_cam.
- If a flight-gradient label is needed, use a small red label: "via u_t, x_{t+1}" near the existing valid upward red arrow to u_t/f_dyn.
- The dashed red stopped-gradient annotation near EMA should remain: "flight-only: sg[c*], theta_cam fixed".

Required result inside the mask:
- No solid red arrow from L_flight into c*_{t+1} or camera branch.
- The legitimate solid red L_flight gradient should remain directed to u_t / f_dyn / x_{t+1}.
- Preserve polished journal style and clean white background.

Avoid:
- Do not redraw the whole figure.
- Do not move boxes.
- Do not introduce new variables.
- Do not damage black forward arrows.
