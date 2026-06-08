Use case: scientific-educational
Task type: edit
Asset type: final SCI journal method figure for "Differentiable Active-Depth Model"

Primary request:
Edit the provided v15 figure with a tiny right-side output-flow correction only. Preserve the left input stack, S_phi rows, pictorial thumbnails, labels, and overall style exactly as much as possible.

Fix only the right output composition:
- Keep Q -> soft validity M -> straight-through mask M_ST.
- M_ST should feed both observed outputs.
- Show D_obs = D_raw x M_ST using a separate small gray input label "D_raw (fixed depth)" feeding the observed-depth output. This D_raw line must NOT appear to originate from the quality Q box.
- Show Q_obs = Q x M_ST using a separate small gray input label "Q (quality)" feeding the observed-quality output.
- Remove or reroute any gray line that visually leaves the quality Q box and goes directly to D_obs.
- The quality Q box should feed soft validity M through the black downward arrow, and may also feed Q_obs as the quality input. It must not directly create D_obs.

Do not change:
- Left active stereo/geometry/camera-register input stack.
- The five S_phi factor rows and their thumbnails.
- The all-five-rows-to-Q combiner.
- Equations and labels except the tiny right-side bypass labels.

Strict exclusions:
- No red arrows.
- No gradient/backward path.
- No loss box.
- No dense formulas.
- No new decorative elements.
