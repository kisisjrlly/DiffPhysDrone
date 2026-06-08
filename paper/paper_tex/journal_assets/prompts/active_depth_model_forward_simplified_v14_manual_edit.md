Use case: scientific-educational
Task type: edit
Asset type: final SCI journal method figure for "Differentiable Active-Depth Model"

Primary request:
Make only data-flow corrections to the provided v13 figure. Preserve the current polished style and pictorial thumbnails. The figure should explain the forward active-depth sensor model only, with no gradients.

Critical left-side correction:
Redraw the left input routing so that geometry does NOT feed into camera registers.
- There must be NO arrowhead touching the left side of the "Camera registers c=[P,E,G]" box.
- There must be NO arrow from the active stereo sketch into the camera-register box.
- There must be NO arrow from "raw depth D_raw" into the camera-register box.
- There must be NO arrow from "scene mask Omega" into the camera-register box.
- Camera registers c=[P,E,G] should have outgoing arrows only, feeding S_phi / relevant factor rows.
- Draw D_raw and Omega as independent geometry/context inputs into S_phi. If the camera-register box blocks the straight path, route D_raw and Omega with clean elbow arrows around the camera-register box into the left edge of the S_phi block.
- Draw speed ||v_t|| as a small context input into the motion blur row only.

Correct sensor factor dependencies:
Inside S_phi, the five rows should read:
1 active return: (P, D_raw)
2 ambient washout: (E, Omega)
3 motion blur: (E, ||v_t||)
4 gain/noise: (G)
5 edge difficulty: (D_raw)
All five rows must visually merge into "quality Q". Use a bracket or common combiner line on the right of the five rows, then a single arrow to quality Q.

Critical right-side correction:
Do not show quality Q directly producing D_obs.
The correct output composition is:
- Q -> soft validity M -> straight-through mask M_ST
- M_ST feeds both observed outputs
- D_obs = D_raw x M_ST
- Q_obs = Q x M_ST
So draw a thin gray bypass from D_raw toward D_obs, and a thin gray bypass from Q toward Q_obs if needed. The mask M_ST should be the main gate feeding both outputs.

Text corrections:
- Keep "Omega in [0,1]: scene/material mask".
- Keep "M = sigmoid((Q - q0) / sigma_q)".
- Keep "Q = clip(Q0 + Omega C_rho, 0,1)".
- Keep "D_obs = D_raw x M_ST" and "Q_obs = Q x M_ST".

Do not add:
- No red arrows.
- No dashed gradient paths.
- No loss box.
- No dense formulas.
- No extra explanatory paragraphs.
- No generic replacement icons; keep the current wall, depth, washout, blur, noise, and edge thumbnails.

Quality requirements:
All arrows must have clear arrowheads and must not terminate on the wrong module. Keep labels readable, aligned, and suitable for a top SCI/CS journal figure.
