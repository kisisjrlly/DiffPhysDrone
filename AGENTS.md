# DiffPhysDrone current research line

- Active work belongs on `active-sensing-grayscale-imx900`. Read
  `docs/grayscale_imx900/CODEX_IMPLEMENTATION_GUIDE.md` before changing the
  method. Legacy D455 branches are historical; do not choose a branch merely
  because its name contains "minimal" or merge legacy camera code here.
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
