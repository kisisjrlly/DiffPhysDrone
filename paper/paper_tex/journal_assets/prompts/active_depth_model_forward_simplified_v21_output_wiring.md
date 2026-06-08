Use case: scientific-educational
Task type: edit
Asset type: final SCI journal method figure for "Differentiable Active-Depth Model"

Primary request:
Make a tiny right-side wiring correction to the provided v20 figure. Preserve all labels, left-side inputs, S_phi rows, thumbnails, and overall style.

Keep the revised paper semantics:
- S_phi outputs only (D_obs, Q_obs).
- No M, M_ST, valid_prob, soft validity map, gradients, or loss boxes.
- Internal gate: V = 1[Q > q0].
- D_obs = D_raw x V.
- Q_obs = Q x V.

Only fix the right output wiring:
- D_raw should feed only the observed depth D_obs box.
- Q should feed the quality gate V and also feed only the observed quality Q_obs box.
- V should feed both D_obs and Q_obs.
- Do not let the D_raw line connect to Q_obs.
- Do not let the D_obs box connect to Q_obs.
- Make the two outputs visually parallel:
  upper output receives D_raw + V
  lower output receives Q + V

Suggested visual:
- Place a small label "D_raw" beside a short arrow entering D_obs from the left.
- Place a small label "Q" beside a short arrow entering Q_obs from the left.
- Place "quality gate V" between Q and the two outputs, with a split branch to both outputs.
- Use clean black/dark-gray arrows with clear arrowheads and no crossing.

Do not change anything else.
