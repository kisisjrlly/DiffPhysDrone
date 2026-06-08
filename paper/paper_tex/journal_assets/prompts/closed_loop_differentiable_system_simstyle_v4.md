Use case: scientific-educational
Task type: generate
Asset type: SCI journal method figure, 1536x512

Primary request:
Draw a polished time-unrolled computational graph for the "Closed-Loop Differentiable System" of a differentiable active-depth quadrotor learning paper. Use the visual style of a differentiable simulation paper: clean horizontal timeline, small depth-image thumbnails, black forward arrows, red backward-gradient arrows, compact legend, minimal variables.

Canvas and style:
1536x512 pixels. White background with subtle pale blue time-step panels. Elegant vector-like academic schematic. Similar to a differentiable-simulation unrolled policy-dynamics figure: repeated compact modules over time, black forward arrows, red backward arrows, small loss symbols below each time step, legend at right. Use rounded rectangles and circular state/action nodes. Keep text large and minimal.

Core forward recurrence:
For each time step k, show:
1. State node: "x_k, c_k".
2. Differentiable perception block: "S_phi" receives "x_k, c_k" and outputs a small grayscale depth-image thumbnail labeled "D_obs,k".
3. The depth thumbnail "D_obs,k" must feed directly into the policy block.
4. Policy block: "Pi_theta" with two small internal branches or output labels:
   - "Pi_f -> u_k"
   - "Pi_c -> c*_{k+1}"
5. Action "u_k" feeds the dynamics block: "f_dyn(x_k,u_k)".
6. Camera target "c*_{k+1}" feeds a small orange "EMA" block together with "c_k", producing "c_{k+1}".
7. Dynamics outputs "x_{k+1}".
8. The next state node is "x_{k+1}, c_{k+1}".
Repeat the same layout for k=0, k=1, ..., k=T-1 in a compact unrolled chain.

Critical accuracy:
- "D_obs,k" must visibly connect to "Pi_theta".
- "Pi_f" and "Pi_c" are two output heads inside the same policy, not two sequential policies.
- "Pi_f" produces only "u_k".
- "Pi_c" produces only "c*_{k+1}".
- "EMA" updates only the camera state "c_{k+1}".
- "f_dyn" updates only the UAV state "x_{k+1}".
- The next recurrent state combines both "x_{k+1}" and "c_{k+1}".
- Do not show Q_t or M_t anywhere.
- Do not show a separate R renderer; keep the figure focused on S_phi and D_obs.

Backward gradients:
Use red arrows:
- From "L_flight" below each dynamics block back through "f_dyn", "u_k", "Pi_theta", "D_obs,k", and "S_phi".
- From "L_sens / L_teach" below each perception block back to "S_phi" and "c_k".
- Add one compact red label near the camera-state gradient: "dL/dc_k".
- Keep backward arrows visually clean and do not overcrowd.

Loss nodes:
Use compact loss symbols below each time step:
"L_flight" near f_dyn and x_{k+1}
"L_sens / L_teach" near S_phi and c_k

Legend:
Right side legend:
black arrow = forward
red arrow = backward
u_k = action
x_k,c_k = UAV/camera state
D_obs = depth observation
Pi_theta = split policy

Text labels to use exactly:
"x_0,c_0", "x_1,c_1", "x_k,c_k", "x_T,c_T"
"S_phi"
"D_obs,0", "D_obs,1", "D_obs,k"
"Pi_theta"
"Pi_f -> u_k"
"Pi_c -> c*_{k+1}"
"EMA"
"f_dyn(x_k,u_k)"
"L_flight"
"L_sens / L_teach"
"dL/dc_k"

Visual detail:
Represent D_obs as small grayscale depth-map thumbnails, like tiny image patches with dark wall/bright aperture patterns. Put the thumbnail above or immediately before Pi_theta, clearly connected by a black arrow. Use pale cyan for S_phi, pale green for Pi_theta, pale orange for EMA, pale purple for f_dyn, red for gradient paths.

Avoid:
Tiny text, long equations, Q_t, M_t, separate R renderer, wrong policy relation, policy-to-policy arrows, EMA feeding f_dyn, f_dyn feeding EMA, missing D_obs-to-policy arrow, missing camera-state recurrence, decorative drone icons, dark background, 3D perspective, cluttered crossings, watermark, logo.
