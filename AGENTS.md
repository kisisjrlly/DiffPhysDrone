# DiffPhysDrone current research line

- Active work belongs on `active-sensing-grayscale-imx900`. Read
  `docs/grayscale_imx900/CODEX_IMPLEMENTATION_GUIDE.md` before changing the
  method. Legacy D455 branches are historical; do not choose a branch merely
  because its name contains "minimal" or merge legacy camera code here.
- User-authorized joint-learning research is isolated on
  `codex/imx900-intent-closed-loop`, based on `d989ad9`. Read
  `docs/grayscale_imx900/INTENT_CLOSED_LOOP.md` for this method. The frozen
  backbone protocol below still governs the historical full/detached study;
  it is not a prohibition on the explicitly requested new joint architecture.
- The primary joint-learning candidate is now `sparse_poles`, documented in
  `docs/grayscale_imx900/SPARSE_POLES_PROTOCOL.md`. Keep the failed double-gate
  pilot as historical evidence. Do not treat a gradient cut as an RL baseline
  or claim method superiority before matched, trained method comparisons.
  Current user scope is full vs nondifferentiable sensing only, cutting both
  E/G→image and pose/motion→image in the latter. Do not add an RL baseline now.
  A shared flight-only imitation warm start must be disclosed, costed and
  checkpoint-matched; it must not label camera settings.
- Current iteration evidence and the two-pole parallax task are in
  `docs/grayscale_imx900/SPARSE_POLES_ITERATION_2026_10_21.md`.
  Do not call the three-pole pilots successful sensing/motion coordination.
  `forced_probe` is an external-action diagnostic, never learned flight.
  In parallax imitation, the initial 24 measurement actions are unlabelled;
  geometry-aware flight labels start only after the informative view change.
- The verified local interpreter is
  `/home/zhaoguodong/miniconda3/envs/mappo-mpc/bin/python` (PyTorch/CUDA).
  Check it before reporting missing dependencies from a system Python.
- The policy sees grayscale frames, not simulator depth or masks. The primary
  camera action is exposure/gain; navigation loss supervises camera learning.
- First establish fixed-grayscale navigation against independently trained
  blind and same-checkpoint zero-image controls. Do not claim camera-learning
  success from a gradient wiring test or a small pilot.
- Full/detached comparisons must keep training/checkpoint/budget matched and
  cut only exposure/gain at the camera boundary. Use paired evaluation seeds.
- The calibration JSON is provisional until measured. Keep runtime results and
  hardware claims separate. See `docs/grayscale_imx900/REVIEW_2026_09_24.md`
  for the legacy-branch correction and current evidence.
