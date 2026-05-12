# Journal Figure Captions

Use the PDF files for manuscript layout. SVG files keep all text editable for
final artwork. The older `paper_assets` directory contains diagnostic plots and
should not be used in a submission.

## Figure 1 | Differentiable active depth sensing for slit navigation.

**a,** Single-wall slit navigation benchmark with randomized slit locations and
three sensor-degradation regimes: glare, low-reflectance dark material, and
specular false depth. **b,**
Closed-loop active-depth formulation. Camera power, exposure, and gain are
policy-controlled variables that change the next depth observation through a
differentiable sensor model. **c,** Training and evaluation protocol. Online
states are relabeled by a differentiable camera teacher, the camera head is
pretrained on these relabeled targets, and flight-control layers are then
adapted while the camera branch is fixed.

## Figure 2 | Training curves show that comparison policies reached stable regimes.

WandB training exports are plotted for the final active-camera policy, fixed
camera, random fixed camera, non-differentiable learned camera, and blind
zero-depth control. **a,** training loss. **b,** training success rate. **c,**
training collision rate. The curves are convergence diagnostics; all navigation
claims use the held-out closed-loop evaluations summarized in Figure 3 and
Tables 1--2.

## Figure 3 | Navigation gains are scene-dependent.

All methods are evaluated for 300 episodes, with 100 episodes in each scene.
**a,** Per-scene success change relative to fixed camera. **b,** Empirical
distribution of terminal goal distance. The proposed policy improves navigation
success while reducing the fraction of episodes that terminate far from the
goal.

## Figure 4 | The learned camera policy implements scene-specific near-slit sensing.

**a,** Exposure-gain response plane, where marker area scales with power and
the grey cross denotes the nominal camera setting. **b,** Exposure and gain
profiles as a function of local distance to the wall; grey shading denotes the
near-slit window. **c,** Median successful trajectories with 10--90% episode
envelopes. Low-reflectance dark-material scenes keep exposure/gain high near
the wall, whereas glare suppresses both parameters.

## Figure 5 | Camera control changes what the policy observes near the slit.

Matched-pose qualitative depth sequences are rendered by
`tools/export_journal_depth_sequences.py` using a far-right slit. The first row
shows the local map, current pose, camera frustum and complete final-policy
trajectory from start through the slit toward the goal. The final policy
trajectory provides the reference poses. At each pose, raw geometric depth is
shown together with observed depth re-rendered using camera parameters from
fixed, random-fixed, and final active-camera policies. The comparison isolates
the sensor-parameter effect on the depth image at identical vehicle poses.
The manuscript uses the glare, dark, and specular composites as a compact
three-subfigure layout.

## Figure 6 | Camera relabeling and flight adaptation are complementary.

Glare-dark camera separation is measured in the relabeled teacher data and in
online closed-loop rollouts for the pretrained, DAgger-relabelled and final
policies.
