# Journal Qualitative Depth Sequences

- scenes: `glare, dark, specular`
- slit slot: `far_right`
- rows: `90`

Outputs:

- `figures/fig5_depth_observation_sequence_<scene>.pdf`: matched-pose raw/depth comparison.
- `figures/extended_data_fig3_method_depth_sequences_<scene>.pdf`: method-own rollout depth sequence.
- `qualitative_depth/depth_sequence_rows.csv`: per-panel camera parameters and local metrics.
- `qualitative_depth/depth_sequence_arrays.npz`: raw depth, observed depth, quality, invalid and effect arrays.

Interpretation:

The matched-pose figure uses the final policy trajectory as the reference pose sequence and re-renders
the sensor observation at those exact poses with camera settings taken from fixed, random-fixed,
and active-camera policies.
The first pose row overlays the complete final-policy local trajectory from start through the slit toward the goal.
This isolates the camera-parameter effect on depth observations. The method-own figure shows what each
policy actually observes along its own rollout.
