Use case: scientific-educational
Task type: generate
Asset type: polished SCI journal method figure, 1536x512

Primary request:
Create a polished method diagram for the "Closed-Loop Differentiable System" subsection of a robotics / computer science paper. The figure must show both forward rollout and backward gradient flow for a differentiable active-depth quadrotor learning system.

Design principle:
Make one central time-step block large and clear, then show temporal recurrence to the next state. Do not overcrowd the figure with three full repeated blocks. Use a single detailed block for time t and a compact ghost block for time t+1 to show iteration.

Canvas:
1536x512 pixels, white background, wide landscape, crisp vector-like style, top-tier IEEE / Science Robotics look.

Main forward path, black arrows:
Left state node:
"x_t, c_t"
and below it a small camera-register label:
"c_t=[P_t,E_t,G_t]"

Central detailed time-step block:
1. "R(x_t,E)" in a pale blue geometric-rendering box, output "D_raw,t".
2. "S_phi" in a larger pale cyan differentiable sensor box. Inputs into this box: "D_raw,t", "c_t", "rho_t", "x_t". Outputs: "D_obs,t", "Q_t", "M_t".
3. "Pi_theta" in a pale green split-policy box. Inside the box, show two small sublabels: "Pi_f" and "Pi_c". Inputs: "D_obs,t", "s_t", "m_t", "h_t", "q_t", "c_t".
4. Policy outputs two branches:
   - "u_t" goes to "f_dyn".
   - "c*_{t+1}" goes to "EMA".
5. "EMA" in pale orange computes "c_{t+1}".
6. "f_dyn" in pale purple computes "x_{t+1}".
Right next-state node:
"x_{t+1}, c_{t+1}".
Then show a faint repeated compact block or arrow labeled "repeat for t+1 ...".

Losses:
Place two large loss boxes below the central block:
1. "L_sens / L_teach" below S_phi, connected to "Q_t, M_t, c_t".
2. "L_flight" below f_dyn and Pi_theta, connected to "x_{t+1}, u_t".

Backward gradients, red arrows:
Use red curved arrows with arrowheads pointing backward.
- From "L_sens / L_teach" back to "S_phi" and to "c_t=[P_t,E_t,G_t]". Label one red arrow "dL/dc_t".
- From "L_flight" back to "f_dyn", then to "u_t", then to "Pi_f" inside "Pi_theta", then to "D_obs,t", then to "S_phi".
- Do NOT draw red gradients passing through "R(x_t,E)" to the environment. R is forward geometric rendering only; S_phi is the differentiable perception path.

Legend:
Small legend in the upper-right:
"black: forward"
"red: backward"
"blue: perception"
"green: policy"
"purple: dynamics"

Style:
Make the diagram elegant and expressive, not plain rectangles. Use rounded boxes, subtle gradients or light hatching inside modules, clean arrow routing, small circular split/fusion nodes for policy outputs, and a thin time-loop arrow from the next-state node back to the next block. Keep all labels large and legible.

Exact labels to use:
"x_t, c_t"
"c_t=[P_t,E_t,G_t]"
"R(x_t,E)"
"D_raw,t"
"S_phi"
"D_obs,t, Q_t, M_t"
"Pi_theta"
"Pi_f"
"Pi_c"
"u_t"
"c*_{t+1}"
"EMA"
"f_dyn"
"x_{t+1}, c_{t+1}"
"L_sens / L_teach"
"L_flight"
"dL/dc_t"
"repeat"

Accuracy constraints:
- The active-depth sensor is S_phi, not R.
- S_phi produces D_obs, Q, and M.
- Pi_theta contains both Pi_f and Pi_c.
- Pi_f outputs u_t.
- Pi_c outputs c*_{t+1}.
- EMA updates c_{t+1}.
- f_dyn updates x_{t+1}.
- Two losses must be visible and distinct.
- Red backward gradients must be visible and must reach c_t through S_phi.
- The time iteration must be visible through x_t,c_t -> x_{t+1},c_{t+1} -> repeat.

Avoid:
Crowded repeated blocks, tiny text, paragraphs, long equations, misspelled variables, "S_ph", "Piheta", missing c_t, missing EMA, missing f_dyn, missing Pi_c, red gradient through R to environment geometry, drone icons, dark background, 3D perspective, watermark, logo.
