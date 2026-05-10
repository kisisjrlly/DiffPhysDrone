# Journal Assets

This directory contains submission-oriented figures and LaTeX tables generated
from:

`paper/experiment/results/final_dagger_flightonly_eval_20260510_splitstem`

The older `paper_assets` directory is diagnostic only and should not be used in a
paper submission. This `journal_assets` directory is the current recommended
figure/table set.

## Main figures

- `figures/fig1_system_protocol.pdf`
- `figures/fig2_training_convergence.pdf`
- `figures/fig3_navigation_performance.pdf`
- `figures/fig4_active_camera_mechanism.pdf`
- `figures/fig5_depth_observation_sequence_glare.pdf`
- `figures/fig5_depth_observation_sequence_dark.pdf`
- `figures/fig5_depth_observation_sequence_specular.pdf`
- `figures/fig6_dagger_relabel_diagnosis.pdf`

## Extended data

- `figures/extended_data_fig1_full_matrix.pdf`
- `figures/extended_data_fig2_terminal_distance.pdf`
- `figures/extended_data_fig3_method_depth_sequences_glare.pdf`
- `figures/extended_data_fig3_method_depth_sequences_dark.pdf`
- `figures/extended_data_fig3_method_depth_sequences_specular.pdf`

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
