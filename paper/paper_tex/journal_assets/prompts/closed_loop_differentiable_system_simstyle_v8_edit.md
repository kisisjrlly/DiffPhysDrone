Use case: scientific-educational
Task type: edit
Asset type: final corrected SCI journal time-unrolled differentiable-system figure, 1536x512

Edit target invariants:
- Preserve the current v7 layout, style, repeated panels, depth thumbnails, state circles, policy blocks, dynamics blocks, EMA blocks, losses, and legend.
- Preserve the clear next-state merge circles.
- Do not add Q_t or M_t. Do not add a separate R renderer.

Critical correction: remove misleading red gradient into EMA
- Remove every solid red arrow from "L_flight" to the orange "EMA" block.
- L_flight should NOT directly update EMA or the camera branch.
- Keep L_flight red gradients only toward:
  f_dyn, u_k, Pi_f / Pi_theta, D_obs,k, and S_phi.
- Keep the dashed red "available / frozen" indication near the Pi_c camera head if present, but it must be dashed and not look like a solid update.

Camera update clarity:
- Keep the black forward arrow from "Pi_c -> c*_{k+1}" into EMA.
- Add or strengthen a black forward arrow from the current state circle's camera component c_k into EMA.
- EMA output remains c_{k+1}, then merges with x_{k+1} from f_dyn at the next-state merge.

Dynamics update clarity:
- f_dyn receives x_k and u_k.
- f_dyn outputs x_{k+1}.
- Do not make EMA feed f_dyn.
- Do not make f_dyn feed EMA.

Perception-to-policy path:
- Keep S_phi -> D_obs,k -> Pi_theta as a clear black forward path.
- D_obs,k remains the grayscale thumbnail.

Backward gradient semantics:
- L_sens / L_teach has red gradients to S_phi and c_k, with label dL/dc_k.
- L_flight has red gradients to f_dyn and Pi_f path only.

Avoid:
Solid red arrows into EMA, solid red arrows into Pi_c, missing c_k -> EMA, missing f_dyn output x_{k+1}, changed labels, tiny text, clutter, watermark, logo.
