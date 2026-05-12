# Journal Assets

This directory contains submission-oriented figures and LaTeX tables generated
from:

`paper/paper_tex/origion_results`

The older `paper_assets` directory is diagnostic only and should not be used in a
paper submission. This `journal_assets` directory is the current recommended
figure/table set.

## Composite figures

- `figures/fig5_depth_observation_sequence_glare.pdf`
- `figures/fig5_depth_observation_sequence_dark.pdf`
- `figures/fig5_depth_observation_sequence_specular.pdf`

## Panel figures

The `figures/panels/` directory contains the standalone subfigure assets used by
the manuscript for Figures 1, 2, 3, 4, and 6.

- `figures/panels/fig2a_training_loss.pdf`
- `figures/panels/fig2b_training_success.pdf`
- `figures/panels/fig2c_training_collision.pdf`
- `figures/panels/fig1a_task_schematic.pdf`
- `figures/panels/fig1b_active_depth_loop.pdf`
- `figures/panels/fig1c_relabeled_training_protocol.pdf`
- `figures/panels/fig3c_scene_success_gain.pdf`
- `figures/panels/fig3d_terminal_distance_ecdf.pdf`
- `figures/panels/fig4b_exposure_gain_plane.pdf`
- `figures/panels/fig4d_exposure_gain_profiles.pdf`
- `figures/panels/fig4e_trajectory_envelopes.pdf`
- `figures/panels/fig6a_camera_semantics_progress.pdf`

## Tables

- `tables/table1_primary_navigation.tex`
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
