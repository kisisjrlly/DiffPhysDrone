Use case: scientific-educational
Task type: edit
Asset type: final corrected SCI journal time-unrolled differentiable-system figure, 1536x512

Edit target invariants:
- Preserve the current v6 layout, style, color palette, panels, state circles, depth thumbnails, policy blocks, f_dyn blocks, EMA blocks, losses, and legend.
- Preserve the overall visual similarity to a differentiable-simulation unrolled computational graph.
- Do not add Q_t or M_t. Do not add a separate R renderer.

Specific correction 1: EMA input from current camera state
- In every time-step panel, add or strengthen a clean black forward arrow from the current state circle's camera component c_k into the EMA block.
- This arrow should make it visually clear that EMA has two inputs:
  current camera state c_k
  policy camera target c*_{k+1}
- Keep the existing black arrow from Pi_c -> c*_{k+1} -> EMA.

Specific correction 2: EMA output labels
- In the k=0 panel, the EMA output must be labeled "c_1", not "c_0".
- In the k=1 panel, EMA output should be "c_2".
- In the generic k panel, EMA output should be "c_{k+1}".
- In the k=T-1 panel, EMA output should be "c_T".

Specific correction 3: dynamics output labels
- Keep f_dyn output labels consistent:
  k=0 panel: x_1
  k=1 panel: x_2
  generic k panel: x_{k+1}
  k=T-1 panel: x_T.

Specific correction 4: next-state merge
- The next state circle should be understood as combining the f_dyn output x_{k+1} and the EMA output c_{k+1}.
- Make the merge into the next state circle clean and unambiguous.
- Do not make EMA look like it feeds f_dyn.
- Do not make f_dyn look like it feeds EMA.

Specific correction 5: keep gradient meaning
- Keep solid red gradients from L_flight to f_dyn and Pi_f, not to Pi_c.
- Keep the dashed red "available / frozen" indication near Pi_c if present.
- Keep L_sens / L_teach gradients to S_phi and c_k.

Avoid:
Changing the whole diagram, adding new modules, changing the legend, making text smaller, introducing misspelled symbols, replacing D_obs thumbnails, cluttered arrows, watermark, logo.
