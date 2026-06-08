# Journal Assets

This directory contains submission-oriented figures and LaTeX tables generated
from:

`paper/paper_tex/origion_results`

The older `paper_assets` directory is diagnostic only and should not be used in a
paper submission. This `journal_assets` directory is the current recommended
figure/table set.

## Composite figures

- `figures/depth_observation_sequence_glare.pdf`
- `figures/depth_observation_sequence_dark.pdf`
- `figures/depth_observation_sequence_specular.pdf`

## Panel figures

The `figures/panels/` directory contains the standalone subfigure assets used by
the manuscript for Figures 1, 2, 3, 4, and 6.

- `figures/panels/training_loss.pdf`
- `figures/panels/training_success.pdf`
- `figures/panels/training_collision.pdf`
- `figures/panels/task_schematic.pdf`
- `figures/panels/active_depth_loop.pdf`
- `figures/panels/relabeled_training_protocol.pdf`
- `figures/panels/primary_success_rate.pdf`
- `figures/panels/primary_collision_rate.pdf`
- `figures/panels/primary_depth_fill.pdf`
- `figures/panels/fill_success_coupling.pdf`
- `figures/panels/scene_success_heatmap.pdf`
- `figures/panels/terminal_distance_ecdf.pdf`
- `figures/panels/camera_peg_grouped_bars.pdf`
- `figures/panels/exposure_gain_plane.pdf`
- `figures/panels/glare_dark_separation.pdf`
- `figures/panels/exposure_gain_profiles.pdf`
- `figures/panels/trajectory_envelopes.pdf`
- `figures/panels/camera_semantics_progress.pdf`
- `figures/panels/stage_success_progress.pdf`

## Tables

- `tables/table2_scene_breakdown.tex`
- `tables/table3_camera_response.tex`

## Statistical conventions

- Binary outcomes use Wilson 95% confidence intervals in tables.
- Continuous episode metrics use bootstrap 95% confidence intervals over episodes.
- Camera behavior is averaged within episode and phase before summarizing across episodes.
- Phase windows: before `x < -0.25 m`, near `|x| <= 0.25 m`, after `x > 0.25 m`.

## Scope note

The assets are professionally formatted, but the evidence remains simulation-only
and single-training-seed. Claims in the manuscript should reflect that scope.
