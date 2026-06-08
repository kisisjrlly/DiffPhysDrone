# Journal Qualitative Depth Sequences

- scenes: `glare, dark, specular`
- slit slot: `far_right`
- rows: `90`

Outputs:

- `figures/fig5_depth_observation_sequence_<scene>.pdf`: matched-pose raw/depth comparison.
- `qualitative_depth/depth_sequence_rows.csv`: per-panel camera parameters and local metrics.
- `qualitative_depth/depth_sequence_arrays.npz`: raw depth, observed depth, quality, invalid and effect arrays.
- `qualitative_depth/trajectory_rows.csv`: complete local rollout trajectory for every exported scene and method.
- `qualitative_depth/trajectories_local_xy.npz`: complete local rollout trajectories keyed by scene/method/slot.
- `qualitative_depth/trajectory_<method>_<scene>_<slot>.npy`: compatibility trajectory arrays for downstream plotting.

Interpretation:

The matched-pose figure uses the final policy trajectory as the reference pose sequence and re-renders
the sensor observation at those exact poses with camera settings taken from fixed, random-fixed,
and active-camera policies.
The first pose row overlays the complete final-policy local trajectory from start through the slit toward the goal.
This isolates the camera-parameter effect on depth observations.
