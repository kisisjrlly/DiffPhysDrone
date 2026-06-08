Use case: scientific-educational
Task type: edit
Asset type: corrected SCI journal method figure, 1536x512

Edit target invariants:
- Preserve the overall style, colors, 1536x512 layout, white background, large central time-step block, legend, red backward arrows, and all existing major labels.
- Preserve the variables: x_t, c_t, R(x_t,E), D_raw,t, S_phi, D_obs,t, Q_t, M_t, Pi_theta, Pi_f, Pi_c, u_t, c*_{t+1}, EMA, f_dyn, x_{t+1}, c_{t+1}, L_sens / L_teach, L_flight.

Primary correction:
Fix the forward data flow after the split policy. The dynamics and camera update must be two parallel state-update paths:
1. Pi_f outputs u_t. The black forward arrow from u_t must go into f_dyn. f_dyn outputs x_{t+1}.
2. Pi_c outputs c*_{t+1}. The black forward arrow from c*_{t+1} must go into EMA. EMA outputs c_{t+1}.
3. The final next-state node combines x_{t+1} from f_dyn and c_{t+1} from EMA.

Important:
- Do NOT show EMA feeding f_dyn as the main forward path.
- Do NOT show c_{t+1} as the input to f_dyn.
- f_dyn should receive x_t and u_t, and optionally a small Delta t_t label.
- EMA should receive c_t and c*_{t+1}.
- Keep L_flight attached to f_dyn, x_{t+1}, and u_t.
- Keep L_sens / L_teach attached to S_phi, Q_t, M_t, and c_t.
- Keep red gradients from L_flight back to f_dyn and Pi_f, and from L_sens / L_teach back to S_phi and c_t.
- Do not draw red gradients through R(x_t,E).

Style correction:
Keep the diagram clean and journal-ready. Use smooth, non-crossing arrows. Make the two parallel outputs from Pi_theta visually clear: upper path u_t -> f_dyn, lower path c*_{t+1} -> EMA.

Avoid:
Changing labels unnecessarily, adding new modules, overcrowding, making text smaller, moving losses too far away, missing the next-state node, watermark, logo.
