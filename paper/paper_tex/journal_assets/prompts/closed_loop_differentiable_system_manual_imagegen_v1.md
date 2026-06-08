Use case: scientific-educational
Task type: generate
Asset type: SCI journal method figure for a computer science / robotics paper

Primary request:
Create a clean, publication-ready diagram for the "Closed-Loop Differentiable System" subsection of a paper about differentiable active-depth perception and quadrotor navigation. The figure must clearly show the forward rollout and backward gradient flow across time.

Core idea:
The system extends a differentiable simulation loop by inserting a differentiable active-depth perception module before the policy. The loop contains UAV state, camera state, geometric depth rendering, differentiable sensor response, split flight/camera policy, camera state update, UAV dynamics, and two losses.

Canvas and composition:
1536x512 pixels. Wide landscape. White background. Left-to-right time-unrolled layout with three repeated time states: t, t+1, t+2. Use black arrows for forward rollout and red arrows for backward gradients. Use a small legend in the upper-right corner.

Layout:
Use a horizontal timeline:
State node at time t: "x_t, c_t"
Then a vertical closed-loop block for time t:
1. Geometric rendering: "R(x_t,E)" outputs "D_raw,t".
2. Differentiable active-depth sensor: "S_phi(D_raw,t,c_t,rho_t,x_t)" outputs "D_obs,t, Q_t, M_t".
3. Split policy: "Pi_theta = {Pi_f, Pi_c}" receives "D_obs,t, s_t, c_t, m_t, h_t, q_t" and outputs two branches:
   - "u_t" to dynamics
   - "c*_{t+1}" to camera EMA
4. Camera state update: "c_{t+1}=EMA(c_t,c*_{t+1})".
5. UAV dynamics: "x_{t+1}=f_dyn(x_t,u_t,Delta t_t)".
The outputs combine into the next state node: "x_{t+1}, c_{t+1}".
Repeat the same structure compactly for t+1 and t+2, using ellipsis if needed.

Loss nodes:
Place two large loss nodes below the timeline:
- "L_sens / L_teach" attached to "Q_t, M_t, c_t" and the sensor block "S_phi".
- "L_flight" attached to "x_{t+1}, u_t" and the dynamics block "f_dyn".

Backward gradients:
Use red arrows with arrowheads pointing backward:
- From "L_sens / L_teach" back to "S_phi" and "c_t=[P,E,G]".
- From "L_flight" back through "f_dyn", "u_t", "Pi_f", "D_obs,t", and "S_phi".
- Show that gradients through "S_phi" reach the camera registers "c_t=[P,E,G]".
- Optionally show a small red label: "dL/dc_t".

Important accuracy constraints:
- R is the native geometric renderer. It is mainly a forward renderer. Do not show red gradients passing through R to scene geometry.
- The differentiable perception module is S_phi, not R.
- S_phi takes D_raw, c_t, rho_t, x_t and outputs D_obs, Q, M.
- Pi_theta contains both flight branch Pi_f and camera branch Pi_c.
- Pi_f outputs the flight action u_t.
- Pi_c outputs the next camera target c*_{t+1}.
- EMA updates the camera state c_{t+1}.
- f_dyn updates the UAV state x_{t+1}.
- Camera state is c_t=[P_t,E_t,G_t].
- The figure must show temporal iteration: x_t,c_t -> x_{t+1},c_{t+1} -> x_{t+2},c_{t+2}.

Style:
Computer science SCI top-tier journal style. Vector-like, crisp, elegant. Use expressive but restrained modules: rounded boxes, subtle pastel fills, thin outlines, clean arrows, small recurrent/time badges. Use pale blue for perception/rendering, pale green for policy, pale orange for camera update, pale purple for dynamics, and red for gradients. Make the figure look like a polished method schematic, not a PowerPoint flowchart.

Text labels:
Use only short labels. Keep all text large and horizontal. Use these exact labels where possible:
"x_t, c_t"
"R(x_t,E)"
"D_raw,t"
"S_phi"
"D_obs,t, Q_t, M_t"
"Pi_theta = {Pi_f, Pi_c}"
"u_t"
"c*_{t+1}"
"EMA"
"f_dyn"
"x_{t+1}, c_{t+1}"
"L_sens / L_teach"
"L_flight"
"dL/dc_t"
"forward"
"backward"

Avoid:
Tiny text, long equations, paragraphs, wrong variables, gradient arrows through R to scene geometry, missing camera state update, missing policy camera branch, missing dynamics, extra invented modules, dark background, 3D perspective, drone icons, decorative clutter, watermark, logo.
