Use case: scientific-educational
Task type: generate
Asset type: SCI journal method figure for a robotics / computer science paper
Primary request: Create a clean, publication-ready neural network architecture diagram for a split-stem recurrent policy used in active depth sensing for quadrotor navigation.

Subject:
A split-stem policy architecture with one shared input and two independent branches. The figure should show data flow accurately and clearly, not as decorative art.

Composition/framing:
Wide landscape architecture figure, 3840 x 1280 aspect ratio, white background, centered, aligned rectangular modules, left-to-right data flow. Three horizontal bands:
1. Top band: Flight branch.
2. Middle-left band: Depth encoding.
3. Bottom band: Camera branch.
Use arrows to show data flow. Use a dashed outline around the camera branch to indicate frozen camera-side parameters during final flight-only adaptation.

Exact module structure to show:
- Input: observed depth D_obs.
- Depth encoding: near inverse-depth max-pool channel and far metric-range avg-pool channel.
- Output of depth encoding: Phi_t, two-channel 12 x 16 representation.
- Flight branch:
  - F_f flight visual stem, Conv 32-64-128, AvgPool 3 x 6, output 192-D.
  - flight state s_t through W_s state projection, output 192-D.
  - LayerNorm plus gated visual-state fusion.
  - GRU_f with residual LayerNorm, hidden state h_t to h_{t+1}.
  - action head W_u, output flight command u_t.
- Camera branch:
  - F_c camera visual stem, same layout as F_f, copy initialized but independent.
  - G_c spatial adapter, Conv 2 to 4 to 4, Pool 2 x 3.
  - residual image adapter A_c, combine into z_t^c.
  - camera state c_t and motion descriptor m_t through W_c and W_m embeddings, 24-D plus 24-D.
  - concatenate z_t^c, camera-state embedding, and motion embedding.
  - W_p pre-layer to 96-D.
  - GRU_c, hidden state q_t to q_{t+1}.
  - sigmoid camera head W_o, output c_star_{t+1} = [P,E,G].

Text labels:
Use only short, readable labels. Do not write paragraphs in the figure. Keep all labels large and horizontal. Use exact short labels such as:
"D_obs", "Depth encoding", "Phi_t (2 ch., 12x16)", "F_f stem", "s_t", "Gated fusion", "GRU_f", "u_t", "F_c stem", "G_c adapter", "z_t^c", "c_t, m_t", "W_c, W_m", "W_p", "GRU_c", "c*_{t+1}=[P,E,G]", "frozen during flight-only".

Style/medium:
Vector-like scientific diagram, crisp lines, subtle pastel fills, no 3D perspective, no shadows, no icons unless very minimal. Top-tier IEEE/Science/Nature style.

Color palette:
Depth encoding in light blue. Flight branch in green. Camera branch in orange. Outputs in soft purple. Frozen outline in muted red dashed stroke. Use black or dark gray text and arrows.

Constraints:
- The flight branch and camera branch must be visually separate.
- F_f and F_c must be shown as independent stems, not shared.
- Camera branch must include both F_c camera stem and G_c spatial adapter.
- Camera branch must include both current camera state c_t and motion descriptor m_t.
- Data flow must end in u_t for flight and c*_{t+1}=[P,E,G] for camera.
- Keep the diagram clean and balanced, with enough white space.
- The figure should look like a computer science SCI journal method figure, not a PowerPoint slide.

Avoid:
Tiny text, misspelled labels, extra invented modules, decorative backgrounds, photorealistic scenes, cartoon style, 3D blocks, icons of drones, excessive formulas, paragraphs inside boxes, overlapping arrows, cluttered arrow crossings, watermark, logo.
