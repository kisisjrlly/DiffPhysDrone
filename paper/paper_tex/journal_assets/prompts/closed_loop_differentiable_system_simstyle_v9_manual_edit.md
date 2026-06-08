Use case: scientific-educational
Task type: edit
Asset type: wide SCI journal method figure, 1536x512 PNG
Primary request: Edit the provided v8 diagram into a publication-quality closed-loop differentiable perception-simulation-control figure. Preserve the clean SCI journal visual style, layout density, palette, and polished appearance of the input image, but correct the data-flow and gradient-flow arrows exactly as specified below.

Input images:
- Image 1: the current v8 diagram to edit; use it as the visual/style/layout basis.

Style/medium: clean vector-like scientific diagram, top-tier computer science journal figure, crisp labels, aligned modules, elegant arrows, high readability on a white background.
Composition/framing: keep the same wide horizontal time-unrolled layout as v8. The diagram should show one time step t and the transition to t+1. It should remain visually polished and readable at 1536x512.
Color palette: preserve the existing blue/cyan/green/orange/red palette. Use black arrows for forward rollout and red arrows for backward gradients. Use dashed red only for optional/stopped-gradient paths.

Text and module labels to include, using math-style notation as cleanly as possible:
- left state node: "x_t, c_t"
- geometric renderer: "R(x_t, E)" and "D_raw,t"
- active-depth sensor: "S_phi"
- observed depth image node: "D_obs,t". Use the existing depth-image/photo style from the input as the visual representation of D_obs,t.
- side output near S_phi: "Q_t, M_t" with small subtitle "quality / validity". This should be a side output for losses, not an input to the policy.
- feature/state encoders: "psi_flight, psi_cam" and "s_t, m_t"
- policy box: a single box labeled "Pi_theta". Do NOT split it into two policy boxes. Inside or under it show outputs "u_t, c*_{t+1}".
- dynamics module: "f_dyn" and output "x_{t+1}"
- camera update module: "EMA" and output "c_{t+1}"
- right next-state node: "x_{t+1}, c_{t+1}"
- bottom loss node 1: "L_flight" with small subtitle "ell_nav(x,u)"
- bottom loss node 2: "L_sens / L_teach" with small subtitle "fill(M), quality(Q)"
- small note near the camera/policy branch: "flight-only: sg[c*], theta_cam fixed"
- legend: black = forward rollout; red = backward gradient; dashed red = stopped/optional.

Forward data-flow arrows, must be exact:
1. x_t goes to R(x_t,E), producing D_raw,t.
2. D_raw,t goes to S_phi.
3. c_t, rho_t, and x_t also feed into S_phi as inputs. Do not show x_t or c_t as outputs of S_phi.
4. S_phi outputs D_obs,t, and D_obs,t feeds directly into the single Pi_theta policy box.
5. S_phi also has a side output Q_t, M_t to the sensing/teacher loss area. Q_t, M_t do NOT feed into Pi_theta.
6. x_t and c_t also feed into psi_flight, psi_cam to form s_t, m_t; s_t, m_t feed into Pi_theta.
7. Pi_theta outputs u_t and c*_{t+1} from the same single policy box.
8. u_t goes to f_dyn, and f_dyn outputs x_{t+1}.
9. c*_{t+1} goes to EMA. c_t also goes to EMA. EMA outputs c_{t+1}.
10. x_{t+1} and c_{t+1} merge into the next-state node, then continue to the next time step.

Backward gradient arrows, must be exact:
1. L_flight should send red gradient arrows to x_{t+1}/f_dyn and to u_t/Pi_theta, then continue backward through D_obs,t to S_phi. It should NOT be drawn as a solid red arrow updating the camera branch in flight-only.
2. L_sens / L_teach should send red gradient arrows through Q_t, M_t and S_phi back to camera parameters / teacher candidate c_bar or c_t.
3. Use dashed red near the camera output branch to indicate an available but stopped/frozen gradient in flight-only: "sg[c*], theta_cam fixed".
4. If showing the exposure timing path, make it a subtle dashed optional red arrow labeled "Delta t_t" or omit it if it hurts readability.

Edit target invariants:
- Preserve the overall clean v8 style and visual polish.
- Preserve the wide 1536x512 aspect ratio.
- Preserve the depth-observation image/photo treatment for D_obs,t.
- Keep the figure readable and uncluttered.
- Use one Pi_theta policy box only.
- Do not add separate Pi_f and Pi_c boxes.
- Do not draw S_phi outputting x_t or c_t.
- Do not draw Q_t or M_t feeding into Pi_theta.
- Do not draw a solid L_flight red gradient arrow updating the camera branch during flight-only.

Avoid:
- Avoid illegible tiny text, crowded spaghetti arrows, duplicated arrowheads, or inconsistent notation.
- Avoid adding new variables not listed above.
- Avoid changing the diagram into a rough sketch. It must look like a polished SCI journal figure.
- Avoid decorative gradients, icons, or 3D effects.
