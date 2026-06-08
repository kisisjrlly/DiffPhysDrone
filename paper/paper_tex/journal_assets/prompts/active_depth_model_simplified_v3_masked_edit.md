Use case: scientific-educational
Task type: precise-object-edit
Asset type: final simplified active-depth model figure
Primary request: Make a very small masked correction to v1. Preserve the whole figure, especially the camera registers c=[P,E,G], exactly as it is. Only remove the red backward-gradient arrows that point upward into "speed ||v_t||".

Required correction inside the mask:
- Remove all red backward-gradient arrows pointing to "speed ||v_t||".
- Do not add any red arrow into speed.
- Speed should remain a black forward input only.
- Preserve or redraw a clean red gradient route that bypasses speed and indicates gradients go to "P,E,G" only.
- Keep the red label "gradient to P,E,G" in the loss box.
- Update the gray bottom note if visible to say: "no gradients to D_raw, Omega, speed, or renderer R".

Do not change:
- Do not alter the P, E, G boxes.
- Do not alter the S_phi block, factor chips, Q/M blocks, output images, or legend.
- Do not redraw the whole figure.

Avoid:
- Avoid red arrows to speed, Omega, scene mask, D_raw, or renderer R.
- Avoid covering black forward arrows.
