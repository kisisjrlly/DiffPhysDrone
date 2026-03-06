import numpy as np


class RerunVis:
    """Lightweight async-friendly logger for single-env visualization in two-stage training.

    - Designed to be no-op when disabled or rerun is unavailable.
    - Logs teacher/student phases separately.
    - Keeps overhead low by accepting already-sampled tensors from caller.
    """

    def __init__(self, enabled=False, app_id="DiffPhysDrone", spawn=True):
        self.enabled = enabled
        self._rr = None
        self._paths = {"teacher": [], "student": []}

        if not enabled:
            return

        try:
            import rerun as rr  # type: ignore
            self._rr = rr
            rr.init(app_id, spawn=spawn)
        except Exception as e:
            print(f"[warn] rerun unavailable, visualization disabled: {e}")
            self.enabled = False
            self._rr = None

    def begin_iter(self, iter_idx: int):
        if not self.enabled:
            return
        self._paths["teacher"].clear()
        self._paths["student"].clear()
        self._rr.set_time_sequence("iter", int(iter_idx))

    def _scalar_msg(self, v: float):
        """API compatibility: rerun-sdk uses Scalars, older variants may expose Scalar."""
        rr = self._rr
        if hasattr(rr, "Scalar"):
            return rr.Scalar(float(v))
        return rr.Scalars(float(v))

    def log_step(self, phase: str, step_idx: int, pos, target, depth=None, cam=None, scalars=None):
        if not self.enabled:
            return
        rr = self._rr
        rr.set_time_sequence("step", int(step_idx))

        p = np.asarray(pos, dtype=np.float32).reshape(3)
        t = np.asarray(target, dtype=np.float32).reshape(3)

        self._paths[phase].append(p.tolist())

        rr.log(f"{phase}/drone/pos", rr.Points3D([p], colors=[[0, 255, 0]], radii=[0.03]))
        rr.log(f"{phase}/target/pos", rr.Points3D([t], colors=[[255, 64, 64]], radii=[0.04]))
        rr.log(f"{phase}/drone/path", rr.LineStrips3D([self._paths[phase]], colors=[[64, 200, 255]]))

        if depth is not None:
            d = np.asarray(depth, dtype=np.float32)
            d = np.clip(d, 0.05, 10.0)
            d = (d - 0.05) / (10.0 - 0.05)
            img = (d * 255.0).astype(np.uint8)
            rr.log(f"{phase}/camera/depth", rr.Image(img))

        if cam is not None:
            fov, exp, iso = [float(x) for x in cam]
            rr.log(f"{phase}/camera/fov", self._scalar_msg(fov))
            rr.log(f"{phase}/camera/exposure", self._scalar_msg(exp))
            rr.log(f"{phase}/camera/iso", self._scalar_msg(iso))

        if scalars is not None:
            for k, v in scalars.items():
                rr.log(f"{phase}/metrics/{k}", self._scalar_msg(float(v)))

    def log_train_scalars(self, scalars: dict):
        if not self.enabled:
            return
        rr = self._rr
        for k, v in scalars.items():
            rr.log(f"train/{k}", self._scalar_msg(float(v)))
