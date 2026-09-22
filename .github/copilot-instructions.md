When executing repository commands, use the conda environment `mappo-mpc`.

This branch is **grayscale/IMX900-only**. Do not restore the old D455
differentiable-depth implementation from history unless the user explicitly
asks. The historical D455 work belongs to `active-sensing-4f-tools-2b-core`.

Key rules:

1. Keep `quadsim_cuda.render_depth` only as a generic geometric ray-
   intersection primitive. It is not a sensor output.
2. The policy visual observation is `[current_gray, previous_gray]`.
3. Camera action/state is exactly `[exposure, gain]`.
4. The main differentiability claim is navigation-loss gradient through the
   grayscale camera model to exposure/gain. Geometry need not be differentiable.
5. Main camera supervision should remain task-driven; do not silently add
   hand-designed image-quality losses.
6. Preserve the matched `full` vs `detached` sensor-gradient ablation.
7. If CUDA code changes, rebuild in the active environment with:
   `pip install -e src`.
8. Make small changes with tests. Do not skip the fixed-gray vs blind gate.
9. IMX900 numerical ranges are provisional until real hardware calibration.
10. Read `docs/grayscale_imx900/CODEX_IMPLEMENTATION_GUIDE.md` before
    implementing a new phase.

When assessing the project, be critical and objective rather than assuming a
proposed method is correct.
