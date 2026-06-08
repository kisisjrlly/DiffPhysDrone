Use case: scientific-educational
Task type: precise-object-edit
Asset type: final simplified active-depth model figure
Primary request: Make a small masked correction to the provided simplified active-depth diagram. Preserve the clean v1 layout and style. Correct only the red backward-gradient routing.

Input image: v1 simplified diagram.

Required correction:
- Red backward gradients from "L_sens / L_teach" must go back to the camera registers "c=[P,E,G]" only.
- Do NOT draw any red backward-gradient arrow to "speed ||v_t||".
- Do NOT draw any red backward-gradient arrow to "scene / material mask Omega".
- Do NOT draw any red backward-gradient arrow to "D_raw", native geometry, or renderer R.
- Keep black forward arrows from speed, Omega, D_raw, and P/E/G into S_phi or factor chips.
- Keep the gray note: "geometry and mask treated as fixed in sensor backward" or equivalent.
- Add or preserve a gray note near the bottom: "no gradients to D_raw, Omega, speed, or renderer R".
- Keep the red label "gradient to P,E,G".
- If needed, show one clean red path: L_sens/L_teach -> D_obs/Q_obs -> Q/M -> S_phi -> P,E,G.

Do not redesign the figure. Keep all modules, thumbnails, title, and polished SCI journal style unchanged outside the mask.

Avoid:
- Avoid red arrows into speed, Omega, scene mask, raw depth, or geometry renderer.
- Avoid clutter or extra formulas.
- Avoid moving major boxes.
