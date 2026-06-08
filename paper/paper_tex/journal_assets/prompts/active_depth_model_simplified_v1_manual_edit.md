Use case: scientific-educational
Task type: edit
Asset type: simplified SCI journal method figure for the subsection "Differentiable Active-Depth Model"
Primary request: Edit the provided complex active-depth model figure into a cleaner, less cluttered publication-quality diagram. Preserve the scientific visual style and the left-to-right data-flow idea, but remove most formulas and excessive text. The figure should clearly explain the differentiable sensing principle rather than documenting every equation.

Input images:
- Image 1: original complex active-depth model figure. Use it as the visual and conceptual basis, but simplify substantially.

Style/medium: clean vector-like scientific diagram, top-tier computer science journal style, white background, crisp arrows, compact labels, balanced spacing, readable at paper width.
Composition/framing: wide horizontal figure. Use four main regions from left to right:
1. Inputs / native geometry
2. Differentiable active-depth sensor response
3. Quality and validity formation
4. Observed outputs and gradient path

Required simplified data flow:
- Left input block: "Native geometry renderer R" producing "D_raw". Show a small depth image thumbnail or grayscale depth map.
- Left/upper input block: "camera registers c=[P,E,G]". Three small chips: "P power", "E exposure", "G gain". These feed into the differentiable sensor response.
- Additional side inputs: "scene / material mask Omega" and "speed ||v_t||". They feed into the sensor response.
- Central block: "Differentiable sensor response S_phi".
- Inside or around S_phi, show only five compact factor chips, no long formulas:
  1. "active return" with note "P, range"
  2. "ambient washout" with note "E, Omega"
  3. "motion blur" with note "E, ||v_t||"
  4. "gain / noise" with note "G"
  5. "edge difficulty" with note "depth discontinuity"
- These factors merge into a node labeled "quality Q".
- quality Q goes to "soft validity M".
- Then show a small straight-through mask node labeled "hard forward, soft backward" or "straight-through mask".
- Final outputs on the right:
  - "observed depth D_obs" with a grayscale depth thumbnail
  - "observed quality Q_obs" or simply "Q_obs" with a heatmap thumbnail
- Black arrows represent forward sensing flow.
- Red arrows represent backward gradients from "L_sens / L_teach" through D_obs, Q, M back to camera registers c=[P,E,G].
- Add a small gray note near D_raw / Omega: "geometry and mask treated as fixed in sensor backward". This is important and should be visible but not dominant.

Text to keep short and exact:
- Title: "Differentiable active-depth model"
- "D_raw"
- "Omega"
- "c=[P,E,G]"
- "S_phi"
- "quality Q"
- "soft validity M"
- "D_obs"
- "Q_obs"
- "L_sens / L_teach"
- "gradient to P,E,G"

Edit target invariants:
- Keep the idea of active stereo / IR projector on the left, but make it smaller and less detailed.
- Keep representative depth / quality map thumbnails, but reduce the number of thumbnails to 3 or 4 total.
- Keep the figure visually polished and readable.
- Use black arrows for forward flow and red arrows for gradient flow.
- Preserve the wide aspect ratio.

Avoid:
- Avoid dense equations such as signal-to-noise formulas, inverse-square formulas, sigmoid equations, or long text paragraphs.
- Avoid many small unreadable panels.
- Avoid showing every regime patch as a separate block. If needed, collapse regimes into "scene / material mask Omega".
- Avoid implying that geometry ray intersections are differentiable in this branch. The differentiable backward should go to P,E,G, not to D_raw geometry.
- Avoid arrows from Q or M into the policy; this figure is only the sensor model.
- Avoid decorative gradients or 3D effects.
