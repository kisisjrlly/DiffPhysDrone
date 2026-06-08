Use case: scientific-educational
Task type: edit
Asset type: corrected SCI journal time-unrolled differentiable system figure, 1536x512

Edit target invariants:
- Preserve the overall differentiable-simulation style, time-unrolled layout, pale blue panels, state circles, depth thumbnails, black forward arrows, red backward arrows, legend, and compact loss boxes.
- Preserve the variables and modules: x_k,c_k, S_phi, D_obs,k, Pi_theta, Pi_f, Pi_c, u_k, c*_{k+1}, EMA, f_dyn, L_sens / L_teach, L_flight.
- Do not add Q_t or M_t. Do not add a separate R renderer.

Primary correction:
Make the forward perception-to-policy path unmistakable in every time-step panel:
1. Add a clear black downward arrow from "S_phi" to the depth thumbnail labeled "D_obs,k".
2. Add a clear black arrow from the "D_obs,k" thumbnail into the "Pi_theta" policy block.
3. Keep optional state/context input from the top state node to Pi_theta, but the depth observation must be visually the dominant policy input.

Policy structure correction:
- Pi_theta must remain one split policy block with two heads:
  "Pi_f -> u_k" and "Pi_c -> c*_{k+1}".
- Do not draw Pi_f and Pi_c as sequential policies.
- Pi_f outputs u_k to f_dyn.
- Pi_c outputs c*_{k+1} to EMA.

State update correction:
- f_dyn outputs x_{k+1}.
- EMA outputs c_{k+1}.
- The next state circle combines x_{k+1}, c_{k+1}.
- Do not draw EMA feeding f_dyn.
- Do not draw f_dyn feeding EMA.

Backward gradients:
- Keep red gradients from L_flight back to f_dyn, Pi_theta, D_obs,k, and S_phi.
- Keep red gradients from L_sens / L_teach back to S_phi and c_k.
- Keep the "dL/dc_k" label near the camera-state gradient.

Style:
Keep labels large and clean. Use smooth non-crossing arrows. Preserve the depth thumbnails as grayscale image patches.

Avoid:
Removing the depth thumbnail, hiding the S_phi-to-D_obs arrow, tiny text, new modules, extra equations, cluttered red arrows, Q_t, M_t, R renderer, watermark, logo.
