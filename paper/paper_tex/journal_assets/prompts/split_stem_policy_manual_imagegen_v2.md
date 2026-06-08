Use case: scientific-educational
Task type: generate
Asset type: wide method figure for a SCI robotics / computer science journal paper

Primary request:
Generate a clean, publication-ready neural network architecture diagram for the Split-Stem Policy in a differentiable active-depth quadrotor navigation paper. The diagram must show the actual data flow from the code and manuscript, with two independent recurrent branches: a flight branch and a camera branch.

Composition/framing:
Wide landscape figure, 3:1 ratio, white background, left-to-right data flow, precise aligned rectangular modules, no title at the top. Use three visually separated regions:
1. Left: depth preprocessing.
2. Upper right: flight branch.
3. Lower right: camera branch.
Use large readable labels and consistent arrow thickness. Keep enough white space so it looks like an IEEE Transactions / Science Robotics method figure.

Core data flow:
- Input: D_obs.
- Depth preprocessing converts D_obs into Phi_t with two channels: near inverse-depth max-pool and far metric-range avg-pool. Label output as "Phi_t (2 ch., 12x16)".
- Phi_t splits into two independent stems:
  - F_f flight visual stem in the flight branch.
  - F_c camera visual stem in the camera branch.
- The stems are independent after initialization. Show this visually with two separate boxes. Do not draw them as a shared encoder.

Flight branch, upper band:
- Label the band "Flight branch".
- F_f stem: "Conv 32-64-128, Pool 3x6, 192-D".
- Flight state input s_t enters W_s state projection, output 192-D.
- F_f feature and W_s feature enter "LayerNorm + gated fusion".
- Then GRU_f with residual LayerNorm, hidden state h_t -> h_{t+1}.
- Then action head W_u.
- Output is u_t.
- Important: W_u is a linear action head. Do NOT put "sigmoid" on W_u. The flight action output u_t must not have sigmoid.

Camera branch, lower band:
- Label the band "Camera branch".
- Put a muted red dashed outline around the camera branch, with a small label "frozen during flight-only".
- F_c camera visual stem receives Phi_t and outputs 192-D.
- G_c spatial adapter also receives Phi_t in parallel, not after F_c. G_c is "Conv 2->4->4, Pool 2x3".
- A_c residual image adapter acts on the F_c feature. Combine F_c, A_c, and G_c into "z_t^c".
- Current camera state c_t enters W_c, output 24-D.
- Motion descriptor m_t enters W_m, output 24-D.
- Concatenate z_t^c, W_c(c_t), and W_m(m_t).
- Then W_p pre-layer, output 96-D.
- Then GRU_c, hidden state q_t -> q_{t+1}.
- Then camera head W_o + sigmoid.
- Output is c*_{t+1} = [P,E,G].

Exact labels to use, keep text short:
"D_obs"
"Depth preprocessing"
"near inv-depth max-pool"
"far metric-range avg-pool"
"Phi_t (2 ch., 12x16)"
"Flight branch"
"F_f stem"
"s_t"
"W_s"
"LayerNorm + gated fusion"
"GRU_f"
"W_u"
"u_t"
"Camera branch"
"F_c stem"
"G_c adapter"
"A_c adapter"
"z_t^c"
"c_t"
"m_t"
"W_c"
"W_m"
"concat"
"W_p"
"GRU_c"
"W_o + sigmoid"
"c*_{t+1} = [P,E,G]"
"frozen during flight-only"

Style/medium:
Vector-like scientific diagram, flat 2D, crisp black arrows, subtle pastel fills, high resolution. No photorealism. No 3D perspective. No decorative icons. Use a restrained academic palette: depth preprocessing light blue, flight branch light green, camera branch light orange, output heads soft purple, frozen dashed outline dark red.

Accuracy constraints:
- Do not show any shared visual encoder after Phi_t. F_f and F_c must be separate independent stems.
- G_c must be parallel to F_c and must receive Phi_t directly.
- A_c is a residual image adapter for the F_c feature, not a replacement for F_c.
- W_u must not include sigmoid.
- Only W_o camera head includes sigmoid.
- Camera branch must include both c_t and m_t as separate inputs.
- Camera output must be c*_{t+1} = [P,E,G].
- Keep all labels large enough for a two-column journal figure after scaling.

Avoid:
Small text, invented modules, missing arrows, wrong arrows, extra formulas, paragraphs in boxes, decorative background, gradient blobs, drone icons, cartoon style, 3D blocks, heavy shadows, crowded crossing arrows, watermark, logo, misspelled variables, "sigmoid" on W_u.
