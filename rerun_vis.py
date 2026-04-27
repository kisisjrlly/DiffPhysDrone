import numpy as np


class RerunVis:
    """Lightweight async-friendly logger for single-env visualization in two-stage training.

    - Designed to be no-op when disabled or rerun is unavailable.
    - Logs teacher/student phases separately.
    - Keeps overhead low by accepting already-sampled tensors from caller.
    """

    def __init__(self, enabled=False, app_id="DiffPhysDrone", spawn=True):
        self.enabled = enabled
        self.app_id = app_id
        self._rr = None
        self._paths = {"teacher": [], "student": []}

        if not enabled:
            return

        try:
            import rerun as rr  # type: ignore
            self._rr = rr
            rr.init(app_id, spawn=spawn)
            self._send_default_blueprint()
        except Exception as e:
            print(f"[warn] rerun unavailable, visualization disabled: {e}")
            self.enabled = False
            self._rr = None

    def _build_eval_dashboard(self, rrb, root="/student", name="student"):
        """Build one eval dashboard bound to one exact student entity root."""
        root = str(root).rstrip("/")

        def p(rel_path):
            return f"{root}/{rel_path}"

        motion_row = rrb.Horizontal(
            rrb.TimeSeriesView(origin=p("metrics/speed_mps"), contents=[p("metrics/speed_mps")], name="speed_mps"),
            rrb.TimeSeriesView(origin=p("metrics/angular_speed_rps"), contents=[p("metrics/angular_speed_rps")], name="angular_speed_rps"),
            rrb.TimeSeriesView(origin=p("metrics/thrust_norm_mps2"), contents=[p("metrics/thrust_norm_mps2")], name="thrust_norm_mps2"),
            rrb.TimeSeriesView(origin=p("metrics/accel_norm_mps2"), contents=[p("metrics/accel_norm_mps2")], name="accel_norm_mps2"),
            rrb.TimeSeriesView(origin=p("metrics/dist_to_goal_m"), contents=[p("metrics/dist_to_goal_m")], name="dist_to_goal_m"),
            name=f"{name}_motion_metrics",
        )
        position_row = rrb.Horizontal(
            rrb.TimeSeriesView(origin=p("metrics/pos_x_m"), contents=[p("metrics/pos_x_m")], name="pos_x_m"),
            rrb.TimeSeriesView(origin=p("metrics/pos_y_m"), contents=[p("metrics/pos_y_m")], name="pos_y_m"),
            rrb.TimeSeriesView(origin=p("metrics/pos_z_m"), contents=[p("metrics/pos_z_m")], name="pos_z_m"),
            name=f"{name}_position_metrics",
        )
        scene_row = rrb.Horizontal(
            rrb.TimeSeriesView(origin=p("metrics/scene_effect_mean"), contents=[p("metrics/scene_effect_mean")], name="scene_effect_mean"),
            rrb.TimeSeriesView(origin=p("metrics/glare_quality_mean"), contents=[p("metrics/glare_quality_mean")], name="glare_quality_mean"),
            rrb.TimeSeriesView(origin=p("metrics/glare_invalid_rate"), contents=[p("metrics/glare_invalid_rate")], name="glare_invalid_rate"),
            rrb.TimeSeriesView(origin=p("metrics/sun_los_mean"), contents=[p("metrics/sun_los_mean")], name="sun_los_mean"),
            rrb.TimeSeriesView(origin=p("metrics/hazard_los_mean"), contents=[p("metrics/hazard_los_mean")], name="hazard_los_mean"),
            name=f"{name}_scene_metrics",
        )
        metrics_block = rrb.Vertical(
            motion_row,
            position_row,
            scene_row,
            name=f"{name}_metrics",
        )

        return rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial3DView(
                    origin=root,
                    contents=[
                        p("drone/**"),
                        p("target/**"),
                        p("world/**"),
                    ],
                    name="student_3d",
                ),
                rrb.Spatial2DView(
                    origin=p("camera/depth_aux"),
                    contents=[p("camera/depth_aux")],
                    name="depth",
                ),
                rrb.Vertical(
                    rrb.Spatial2DView(
                        origin=p("camera/quality"),
                        contents=[p("camera/quality")],
                        name="quality",
                    ),
                    rrb.Spatial2DView(
                        origin=p("camera/invalid"),
                        contents=[p("camera/invalid")],
                        name="invalid",
                    ),
                    rrb.Spatial2DView(
                        origin=p("camera/scene_effect"),
                        contents=[p("camera/scene_effect")],
                        name="scene_effect",
                    ),
                    name="scene_maps",
                ),
                rrb.Vertical(
                    rrb.TimeSeriesView(origin=p("camera/power"), contents=[p("camera/power")], name="power"),
                    rrb.TimeSeriesView(origin=p("camera/exposure"), contents=[p("camera/exposure")], name="exposure"),
                    rrb.TimeSeriesView(origin=p("camera/gain"), contents=[p("camera/gain")], name="gain"),
                    name="camera_params",
                ),
                name="top_row",
            ),
            metrics_block,
            name=name,
        )

    def send_eval_episode_blueprint(self, num_episodes: int, vis_episode_idx: int = -1):
        """Send an eval blueprint with one selectable tab per logged episode."""
        if not self.enabled or self._rr is None:
            return
        try:
            import rerun.blueprint as rrb  # type: ignore

            if int(vis_episode_idx) >= 0:
                root = "/student"
                dashboard = self._build_eval_dashboard(rrb, root=root, name=f"ep_{int(vis_episode_idx):03d}")
                bp = rrb.Blueprint(dashboard, auto_layout=False, auto_views=False)
            else:
                n = max(1, int(num_episodes))
                tabs = [
                    self._build_eval_dashboard(
                        rrb,
                        root=f"/episodes/ep_{ep_idx:03d}/student",
                        name=f"ep_{ep_idx:03d}",
                    )
                    for ep_idx in range(n)
                ]
                bp = rrb.Blueprint(
                    rrb.Tabs(*tabs, active_tab=0, name="episodes"),
                    auto_layout=False,
                    auto_views=False,
                )
            self._rr.send_blueprint(bp)
        except Exception as e:
            print(f"[warn] failed to send eval episode blueprint: {e}")

    def _send_default_blueprint(self):
        """Send a deterministic dashboard layout so key metrics are always visible."""
        if not self.enabled or self._rr is None:
            return
        rr = self._rr
        try:
            import rerun.blueprint as rrb  # type: ignore

            if "eval" in str(self.app_id).lower():
                dashboard = self._build_eval_dashboard(rrb, root="/student", name="student")
                bp = rrb.Blueprint(dashboard, auto_layout=False, auto_views=False)
                rr.send_blueprint(bp)
                return
            else:
                metrics_block = rrb.Horizontal(
                    rrb.TimeSeriesView(origin="/train/loss", contents=["/train/loss"], name="loss"),
                    rrb.TimeSeriesView(origin="/train/loss_distill", contents=["/train/loss_distill"], name="loss_distill"),
                    rrb.TimeSeriesView(origin="/train/sim_fps", contents=["/train/sim_fps"], name="sim_fps"),
                    rrb.TimeSeriesView(origin="/train/iter_per_sec", contents=["/train/iter_per_sec"], name="iter_per_sec"),
                    name="train_metrics",
                )

            bp = rrb.Blueprint(metrics_block, auto_layout=False, auto_views=False)
            rr.send_blueprint(bp)
        except Exception as e:
            print(f"[warn] failed to send rerun blueprint, fallback to auto layout: {e}")

    def _compute_scene_bounds(self, max_speed=None, y_stretch=None, scale=None):
        """固定最小验证地图的 AABB。"""
        _ = max_speed, y_stretch, scale
        scene_min = np.array([-5.8, -5.8, -0.5], dtype=np.float32)
        scene_max = np.array([5.8, 5.8, 4.0], dtype=np.float32)
        return scene_min, scene_max

    def _build_box_mesh(self, boxes):
        """Build a flat-shaded triangle mesh for axis-aligned boxes.

        boxes: (N, 6) with [cx, cy, cz, hx, hy, hz]
        returns: (vertex_positions, triangle_indices, vertex_normals)
        """
        b = np.asarray(boxes, dtype=np.float32)
        if b.size == 0:
            return (
                np.empty((0, 3), dtype=np.float32),
                np.empty((0, 3), dtype=np.uint32),
                np.empty((0, 3), dtype=np.float32),
            )

        # 6 faces, 4 vertices per face, 2 triangles per face.
        face_specs = [
            ([1.0, 0.0, 0.0], [[1, -1, -1], [1, 1, -1], [1, 1, 1], [1, -1, 1]]),
            ([-1.0, 0.0, 0.0], [[-1, 1, -1], [-1, -1, -1], [-1, -1, 1], [-1, 1, 1]]),
            ([0.0, 1.0, 0.0], [[-1, 1, -1], [1, 1, -1], [1, 1, 1], [-1, 1, 1]]),
            ([0.0, -1.0, 0.0], [[1, -1, -1], [-1, -1, -1], [-1, -1, 1], [1, -1, 1]]),
            ([0.0, 0.0, 1.0], [[-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]]),
            ([0.0, 0.0, -1.0], [[-1, 1, -1], [1, 1, -1], [1, -1, -1], [-1, -1, -1]]),
        ]

        vertices = []
        normals = []
        triangles = []
        for row in b:
            cx, cy, cz, hx, hy, hz = row.tolist()
            center = np.array([cx, cy, cz], dtype=np.float32)
            half = np.array([hx, hy, hz], dtype=np.float32)
            for normal, local_corners in face_specs:
                base = len(vertices)
                for corner in local_corners:
                    vertices.append((center + half * np.asarray(corner, dtype=np.float32)).tolist())
                    normals.append(list(normal))
                triangles.append([base + 0, base + 1, base + 2])
                triangles.append([base + 0, base + 2, base + 3])

        return (
            np.asarray(vertices, dtype=np.float32),
            np.asarray(triangles, dtype=np.uint32),
            np.asarray(normals, dtype=np.float32),
        )

    def begin_episode(self, ep_idx: int, step_base: int = 0):
        """Reset all per-episode data in rerun at the start of a new episode.

        Clears 3D scene entities, flight path, camera params, and metrics so
        the viewer only shows the current episode's data.
        """
        if not self.enabled or self._rr is None:
            return
        rr = self._rr
        self._paths["teacher"].clear()
        self._paths["student"].clear()

        # Use a monotonically increasing step timeline across eval episodes.
        # If every episode reuses step=0..T on the same scalar paths, Rerun
        # overlays/joins multiple episode traces and plots look thick/zigzag.
        try:
            rr.set_time_sequence("step", int(step_base))
        except Exception:
            pass

        # Clear all student entities (3D scene, path, metrics, camera params)
        for ns in ("student", "teacher"):
            try:
                rr.log(ns, rr.Clear(recursive=True))
            except Exception:
                pass

        try:
            rr.set_time_sequence("iter", int(ep_idx))
        except Exception:
            pass

    def begin_iter(self, iter_idx: int, reset_scene: bool = False, step_base: int = 0):
        if not self.enabled or self._rr is None:
            return
        self._paths["teacher"].clear()
        self._paths["student"].clear()
        try:
            self._rr.set_time_sequence("step", int(step_base))
        except Exception:
            pass
        if reset_scene:
            rr = self._rr
            # 评估多 episode 时，清空上一轮实体，避免 viewer 中残留旧轨迹/障碍显示。
            try:
                rr.log("student", rr.Clear(recursive=True))
            except Exception:
                pass
            try:
                rr.log("teacher", rr.Clear(recursive=True))
            except Exception:
                pass
        self._rr.set_time_sequence("iter", int(iter_idx))

    def log_environment(self, phase: str,
                        balls=None, voxels=None, cyl=None, cyl_h=None,
                        start=None, target=None,
                        scene_name=None, scene_effects=None,
                        max_speed=None, y_stretch=None, scale=None,
                        step_idx: int = 0):
        """Log one environment snapshot for global 3D inspection.

        All arrays should already be numpy arrays for a single env index:
          balls:  (N,4) [x,y,z,r]
          voxels: (M,6) [x,y,z,rx,ry,rz]
          cyl:    (K,3) [x,y,r]      (approx rendered as points)
          cyl_h:  (L,3) [x,z,r]      (approx rendered as points)
          start/target: (3,)
        
        Environment info (optional, for accurate AABB):
          max_speed: 最大飞行速度（用于推断场景大小）
          y_stretch: Y轴拉伸系数（前后距离拉伸比例）
          scale:     X轴场景缩放系数
        """
        if not self.enabled or self._rr is None:
            return
        rr = self._rr
        # Use a deterministic step for static scene entities in current iter.
        try:
            rr.set_time_sequence("step", int(step_idx))
        except Exception as e:
            print(f"[rerun warn] failed to set time sequence: {e}")
            return

        # 环境参数给出“理论范围”（仅作兜底），主AABB优先由当前实体数据决定。
        env_min, env_max = self._compute_scene_bounds(
            max_speed=max_speed,
            y_stretch=y_stretch,
            scale=scale,
        )
        scene_min = np.array([np.inf, np.inf, np.inf], dtype=np.float32)
        scene_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float32)
        has_bounds = False
        
        # Estimate scene scale for readable world axes.
        axis_len = 3.0

        def _expand_bounds(lo, hi):
            """安全地扩展AABB，避免NaN或极端值。"""
            nonlocal scene_min, scene_max, has_bounds
            lo = np.asarray(lo, dtype=np.float32)
            hi = np.asarray(hi, dtype=np.float32)
            # 检查是否为有效的有限数值
            if np.all(np.isfinite(lo)) and np.all(np.isfinite(hi)):
                scene_min = np.minimum(scene_min, lo)
                scene_max = np.maximum(scene_max, hi)
                has_bounds = True

        if voxels is not None:
            try:
                v = np.asarray(voxels, dtype=np.float32)
                if v.size > 0 and v.shape[1] >= 6:
                    # 过滤“哨兵/极端”体素（例如 roof 高度 200+、半尺寸 200）
                    center = v[:, :3]
                    half = np.abs(v[:, 3:6])
                    valid = np.isfinite(v).all(axis=1)
                    valid &= (half.max(axis=1) < 80.0)
                    valid &= (np.abs(center).max(axis=1) < 300.0)
                    vv = v[valid]
                    if vv.size > 0:
                        lo = (vv[:, :3] - vv[:, 3:6]).min(0)
                        hi = (vv[:, :3] + vv[:, 3:6]).max(0)
                        _expand_bounds(lo, hi)
            except Exception as e:
                print(f"[rerun warn] failed to expand bounds from voxels: {e}")
                
        if balls is not None:
            try:
                b = np.asarray(balls, dtype=np.float32)
                if b.size > 0 and b.shape[1] >= 4:
                    r = b[:, 3:4]
                    lo = (b[:, :3] - r).min(0)
                    hi = (b[:, :3] + r).max(0)
                    _expand_bounds(lo, hi)
            except Exception as e:
                print(f"[rerun warn] failed to expand bounds from balls: {e}")
                
        if cyl is not None:
            try:
                c = np.asarray(cyl, dtype=np.float32)
                if c.size > 0 and c.shape[1] >= 3:
                    c_lo = np.stack([c[:, 0] - c[:, 2], c[:, 1] - c[:, 2], np.full_like(c[:, 0], -1.0)], -1).min(0)
                    c_hi = np.stack([c[:, 0] + c[:, 2], c[:, 1] + c[:, 2], np.full_like(c[:, 0], 2.0)], -1).max(0)
                    _expand_bounds(c_lo, c_hi)
            except Exception as e:
                print(f"[rerun warn] failed to expand bounds from cyl: {e}")
                
        if cyl_h is not None:
            try:
                ch = np.asarray(cyl_h, dtype=np.float32)
                if ch.size > 0 and ch.shape[1] >= 3:
                    ch_lo = np.stack([ch[:, 0] - ch[:, 2], np.full_like(ch[:, 0], -1.0), ch[:, 1] - ch[:, 2]], -1).min(0)
                    ch_hi = np.stack([ch[:, 0] + ch[:, 2], np.full_like(ch[:, 0], 1.0), ch[:, 1] + ch[:, 2]], -1).max(0)
                    _expand_bounds(ch_lo, ch_hi)
            except Exception as e:
                print(f"[rerun warn] failed to expand bounds from cyl_h: {e}")

        # 起点终点也应参与边界，避免目标在盒子外。
        if start is not None:
            s = np.asarray(start, dtype=np.float32).reshape(3)
            _expand_bounds(s, s)
        if target is not None:
            t = np.asarray(target, dtype=np.float32).reshape(3)
            _expand_bounds(t, t)

        # 若没有任何有效实体，则回退到理论边界；否则按实体边界添加小边距。
        if not has_bounds:
            scene_min, scene_max = env_min, env_max
        else:
            span_now = np.maximum(scene_max - scene_min, 1e-3)
            margin = np.maximum(0.08 * span_now, np.array([0.8, 0.8, 0.5], dtype=np.float32))
            scene_min = scene_min - margin
            scene_max = scene_max + margin

        span = scene_max - scene_min
        axis_len = float(np.clip(0.12 * np.max(span), 0.55, 1.25))
        scene_diag = float(np.linalg.norm(span))
        # 线宽按场景尺度自适应，但保持上限较小，避免“线框淹没实体面”。
        base_r = float(np.clip(0.00045 * scene_diag, 0.002, 0.008))
        box_r = float(max(0.0015, base_r * 0.8))
        voxel_edge_r = float(max(0.0018, base_r * 0.95))
        aabb_edge_r = float(max(0.0022, base_r * 1.2))

        # 有限长度可视化（真实物理里 cyl / cyl_h 为无限圆柱）。
        # 这里改为“稳健范围估计”，不再直接跟随全局 AABB，
        # 避免存在超大体素时，圆柱显示高度/长度被异常拉长。
        def _robust_axis_range(samples, fallback=(-2.5, 4.5), min_span=4.0, max_span=18.0):
            s = np.asarray(samples, dtype=np.float32).reshape(-1)
            s = s[np.isfinite(s)]
            if s.size < 4:
                return float(fallback[0]), float(fallback[1])
            q25, q50, q75 = np.percentile(s, [25.0, 50.0, 75.0])
            iqr = max(float(q75 - q25), 1e-3)
            span = float(np.clip(4.0 * iqr + 2.0, min_span, max_span))
            lo = float(q50 - 0.5 * span)
            hi = float(q50 + 0.5 * span)
            return lo, hi

        z_samples = []
        y_samples = []
        if start is not None:
            s = np.asarray(start, dtype=np.float32).reshape(3)
            y_samples.append(float(s[1]))
            z_samples.append(float(s[2]))
        if target is not None:
            t = np.asarray(target, dtype=np.float32).reshape(3)
            y_samples.append(float(t[1]))
            z_samples.append(float(t[2]))
        if balls is not None:
            b = np.asarray(balls, dtype=np.float32)
            if b.size > 0:
                y_samples.extend(b[:, 1].tolist())
                z_samples.extend((b[:, 2] - b[:, 3]).tolist())
                z_samples.extend((b[:, 2] + b[:, 3]).tolist())
        if voxels is not None:
            v = np.asarray(voxels, dtype=np.float32)
            if v.size > 0:
                y_samples.extend((v[:, 1] - v[:, 4]).tolist())
                y_samples.extend((v[:, 1] + v[:, 4]).tolist())
                z_samples.extend((v[:, 2] - v[:, 5]).tolist())
                z_samples.extend((v[:, 2] + v[:, 5]).tolist())

        z_lo, z_hi = _robust_axis_range(z_samples, fallback=(-2.5, 4.5), min_span=4.0, max_span=14.0)
        y_lo, y_hi = _robust_axis_range(y_samples, fallback=(-12.0, 12.0), min_span=8.0, max_span=30.0)

        def _build_circle(center_xyz, ex_xyz, ey_xyz, radius, n_seg=24):
            c = np.asarray(center_xyz, dtype=np.float32)
            ex = np.asarray(ex_xyz, dtype=np.float32)
            ey = np.asarray(ey_xyz, dtype=np.float32)
            th = np.linspace(0.0, 2.0 * np.pi, int(n_seg) + 1, dtype=np.float32)
            pts = c[None, :] + radius * (np.cos(th)[:, None] * ex[None, :] + np.sin(th)[:, None] * ey[None, :])
            return pts.tolist()

        # reference world axes
        rr.log(
            f"{phase}/world/axes",
            rr.Arrows3D(
                vectors=[[axis_len, 0.0, 0.0], [0.0, axis_len, 0.0], [0.0, 0.0, axis_len]],
                origins=[[0.0, 0.0, 0.0]] * 3,
                colors=[[255, 80, 80], [80, 255, 80], [80, 160, 255]],
                radii=[0.008, 0.008, 0.008],
                labels=["+X", "+Y", "+Z"],
                show_labels=True,
            ),
        )

        # World AABB overlay for map extent awareness.
        center = ((scene_min + scene_max) * 0.5).astype(np.float32)
        half = ((scene_max - scene_min) * 0.5).astype(np.float32)
        rr.log(
            f"{phase}/world/aabb",
            rr.Boxes3D(
                centers=[center.tolist()],
                half_sizes=[half.tolist()],
                # 降低填充透明度，主要依赖高对比线框强调边界。
                colors=[[255, 240, 120, 24]],
                radii=[box_r],
                show_labels=False,
            ),
        )

        cx, cy, cz = center.tolist()
        hx, hy, hz = half.tolist()
        corners = np.array([
            [cx - hx, cy - hy, cz - hz],
            [cx + hx, cy - hy, cz - hz],
            [cx + hx, cy + hy, cz - hz],
            [cx - hx, cy + hy, cz - hz],
            [cx - hx, cy - hy, cz + hz],
            [cx + hx, cy - hy, cz + hz],
            [cx + hx, cy + hy, cz + hz],
            [cx - hx, cy + hy, cz + hz],
        ], dtype=np.float32)
        edge_idx = [(0, 1), (1, 2), (2, 3), (3, 0),
                    (4, 5), (5, 6), (6, 7), (7, 4),
                    (0, 4), (1, 5), (2, 6), (3, 7)]
        aabb_edges = [[corners[a].tolist(), corners[b].tolist()] for a, b in edge_idx]
        rr.log(
            f"{phase}/world/aabb_edges",
            rr.LineStrips3D(
                aabb_edges,
                colors=[[255, 245, 120, 255]] * len(aabb_edges),
                radii=[aabb_edge_r] * len(aabb_edges),
            ),
        )
        rr.log(
            f"{phase}/world/aabb_corners",
            rr.Points3D(
                corners,
                colors=[[255, 245, 120]] * corners.shape[0],
                radii=[aabb_edge_r * 1.15] * corners.shape[0],
            ),
        )

        if voxels is not None:
            v = np.asarray(voxels, dtype=np.float32)
            if v.size > 0:
                if hasattr(rr, "Mesh3D"):
                    vertex_positions, triangle_indices, vertex_normals = self._build_box_mesh(v)
                    rr.log(
                        f"{phase}/world/voxels_mesh",
                        rr.Mesh3D(
                            vertex_positions=vertex_positions,
                            triangle_indices=triangle_indices,
                            vertex_normals=vertex_normals,
                            albedo_factor=[80, 185, 245, 170],
                        ),
                    )
                else:
                    rr.log(
                        f"{phase}/world/voxels",
                        rr.Boxes3D(
                            centers=v[:, :3],
                            half_sizes=v[:, 3:6],
                            colors=[[60, 200, 255, 170]] * v.shape[0],
                            radii=[box_r] * v.shape[0],
                        ),
                    )

                # Add wireframe edges to make box obstacles stand out.
                edges = []
                for row in v:
                    cx, cy, cz, hx, hy, hz = row.tolist()
                    corners = np.array([
                        [cx - hx, cy - hy, cz - hz],
                        [cx + hx, cy - hy, cz - hz],
                        [cx + hx, cy + hy, cz - hz],
                        [cx - hx, cy + hy, cz - hz],
                        [cx - hx, cy - hy, cz + hz],
                        [cx + hx, cy - hy, cz + hz],
                        [cx + hx, cy + hy, cz + hz],
                        [cx - hx, cy + hy, cz + hz],
                    ], dtype=np.float32)
                    idx = [(0, 1), (1, 2), (2, 3), (3, 0),
                           (4, 5), (5, 6), (6, 7), (7, 4),
                           (0, 4), (1, 5), (2, 6), (3, 7)]
                    for a, b in idx:
                        edges.append([corners[a].tolist(), corners[b].tolist()])
                rr.log(
                    f"{phase}/world/voxels_edges",
                    rr.LineStrips3D(
                        edges,
                        colors=[[20, 170, 255, 255]] * len(edges),
                        radii=[voxel_edge_r] * len(edges),
                    ),
                )

        if balls is not None:
            b = np.asarray(balls, dtype=np.float32)
            if b.size > 0:
                rr.log(
                    f"{phase}/world/balls",
                    rr.Points3D(
                        b[:, :3],
                        radii=b[:, 3],
                        colors=[[255, 180, 80]] * b.shape[0],
                    ),
                )

        # Cylinders are logged as center points + radius proxy for global context.
        if cyl is not None:
            c = np.asarray(cyl, dtype=np.float32)
            if c.size > 0:
                c_pos = np.stack([c[:, 0], c[:, 1], np.full_like(c[:, 0], (z_lo + z_hi) * 0.5)], -1)
                rr.log(
                    f"{phase}/world/cyl",
                    rr.Points3D(c_pos, radii=np.maximum(c[:, 2], 0.02), colors=[[120, 200, 255]] * c.shape[0]),
                )

                cyl_wire = []
                for row in c:
                    cx, cy, rad = [float(x) for x in row.tolist()]
                    z_mid = 0.5 * (z_lo + z_hi)
                    ring_lo = _build_circle([cx, cy, z_lo], [1, 0, 0], [0, 1, 0], rad)
                    ring_hi = _build_circle([cx, cy, z_hi], [1, 0, 0], [0, 1, 0], rad)
                    cyl_wire.append(ring_lo)
                    cyl_wire.append(ring_hi)
                    for a in (0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi):
                        dx = rad * float(np.cos(a))
                        dy = rad * float(np.sin(a))
                        cyl_wire.append([[cx + dx, cy + dy, z_lo], [cx + dx, cy + dy, z_hi]])
                    # 轴线，便于识别“这是沿 z 方向的竖直圆柱”
                    cyl_wire.append([[cx, cy, z_lo], [cx, cy, z_hi]])
                    cyl_wire.append([[cx - rad, cy, z_mid], [cx + rad, cy, z_mid]])
                rr.log(
                    f"{phase}/world/cyl_wire",
                    rr.LineStrips3D(
                        cyl_wire,
                        colors=[[120, 200, 255, 255]] * len(cyl_wire),
                        radii=[voxel_edge_r * 0.9] * len(cyl_wire),
                    ),
                )

        if cyl_h is not None:
            ch = np.asarray(cyl_h, dtype=np.float32)
            if ch.size > 0:
                ch_pos = np.stack([ch[:, 0], np.full_like(ch[:, 0], (y_lo + y_hi) * 0.5), ch[:, 1]], -1)
                rr.log(
                    f"{phase}/world/cyl_h",
                    rr.Points3D(ch_pos, radii=np.maximum(ch[:, 2], 0.02), colors=[[180, 120, 255]] * ch.shape[0]),
                )

                cyl_h_wire = []
                for row in ch:
                    cx, cz, rad = [float(x) for x in row.tolist()]
                    y_mid = 0.5 * (y_lo + y_hi)
                    ring_lo = _build_circle([cx, y_lo, cz], [1, 0, 0], [0, 0, 1], rad)
                    ring_hi = _build_circle([cx, y_hi, cz], [1, 0, 0], [0, 0, 1], rad)
                    cyl_h_wire.append(ring_lo)
                    cyl_h_wire.append(ring_hi)
                    for a in (0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi):
                        dx = rad * float(np.cos(a))
                        dz = rad * float(np.sin(a))
                        cyl_h_wire.append([[cx + dx, y_lo, cz + dz], [cx + dx, y_hi, cz + dz]])
                    # 轴线，便于识别“这是沿 y 方向的水平圆柱”
                    cyl_h_wire.append([[cx, y_lo, cz], [cx, y_hi, cz]])
                    cyl_h_wire.append([[cx - rad, y_mid, cz], [cx + rad, y_mid, cz]])
                rr.log(
                    f"{phase}/world/cyl_h_wire",
                    rr.LineStrips3D(
                        cyl_h_wire,
                        colors=[[180, 120, 255, 255]] * len(cyl_h_wire),
                        radii=[voxel_edge_r * 0.9] * len(cyl_h_wire),
                    ),
                )

        if start is not None:
            s = np.asarray(start, dtype=np.float32).reshape(3)
            rr.log(
                f"{phase}/world/start",
                rr.Points3D(
                    [s],
                    radii=[0.10],
                    colors=[[30, 255, 120]],
                    labels=["START"],
                    show_labels=True,
                ),
            )
        if target is not None:
            t = np.asarray(target, dtype=np.float32).reshape(3)
            rr.log(
                f"{phase}/world/goal",
                rr.Points3D(
                    [t],
                    radii=[0.11],
                    colors=[[255, 70, 70]],
                    labels=["GOAL"],
                    show_labels=True,
                ),
            )

        scene_name = None if scene_name is None else str(scene_name)
        fx = scene_effects or {}
        if scene_name is not None and scene_name.startswith('sun_glare') and 'sun_anchor' in fx:
            sun = np.asarray(fx['sun_anchor'], dtype=np.float32).reshape(3)
            rr.log(
                f"{phase}/world/scene/sun_anchor",
                rr.Points3D([sun], radii=[0.16], colors=[[255, 220, 80]], labels=["SUN"], show_labels=True),
            )
            rr.log(
                f"{phase}/world/scene/sun_rays",
                rr.Arrows3D(
                    origins=[sun.tolist()],
                    vectors=[[-1.5, 0.0, -0.2]],
                    colors=[[255, 220, 80]],
                    radii=[0.025],
                    labels=["SUN_RAY"],
                    show_labels=False,
                ),
            )
        elif scene_name is not None and scene_name.startswith('specular_trap') and 'panel_center' in fx:
            panel_center = np.asarray(fx['panel_center'], dtype=np.float32).reshape(3)
            half_y = float(fx.get('panel_half_y', 0.95))
            half_z = float(fx.get('panel_half_z', 1.15))
            rr.log(
                f"{phase}/world/scene/specular_panel",
                rr.Boxes3D(
                    centers=[panel_center.tolist()],
                    half_sizes=[[0.03, half_y, half_z]],
                    colors=[[255, 180, 120, 120]],
                    radii=[0.004],
                    labels=["SPECULAR_PANEL"],
                    show_labels=True,
                ),
            )
        elif scene_name is not None and scene_name.startswith('vantablack_gap') and 'gap_center' in fx:
            gap_center = np.asarray(fx['gap_center'], dtype=np.float32).reshape(3)
            half_y = float(fx.get('gap_half_w', 0.58))
            half_z = float(fx.get('gap_half_h', 0.95))
            rr.log(
                f"{phase}/world/scene/vantablack_gap",
                rr.Boxes3D(
                    centers=[gap_center.tolist()],
                    half_sizes=[[0.03, half_y, half_z]],
                    colors=[[30, 30, 30, 210]],
                    radii=[0.004],
                    labels=["VANTABLACK_GAP"],
                    show_labels=True,
                ),
            )
        elif scene_name is not None and scene_name.startswith('dark_morphing') and 'slit_center' in fx:
            slit_center = np.asarray(fx['slit_center'], dtype=np.float32).reshape(3)
            half_y = float(fx.get('gap_half_w', 0.32))
            half_z = float(fx.get('gap_half_h', 0.88))
            rr.log(
                f"{phase}/world/scene/dark_slit",
                rr.Boxes3D(
                    centers=[slit_center.tolist()],
                    half_sizes=[[0.03, half_y, half_z]],
                    colors=[[90, 90, 130, 190]],
                    radii=[0.004],
                    labels=["DARK_SLIT"],
                    show_labels=True,
                ),
            )

    def _scalar_msg(self, v: float):
        """API compatibility: rerun-sdk uses Scalars, older variants may expose Scalar."""
        rr = self._rr
        if hasattr(rr, "Scalar"):
            return rr.Scalar(float(v))
        return rr.Scalars(float(v))

    def _img_u8(self, img, mode: str = "depth"):
        """Convert image to uint8 for rerun logging.

        mode:
          - depth: expects metric depth in meters, maps [0.05, 10.0] -> [0, 255]
          - depth_aux: expects metric depth in meters, uses robust per-frame contrast
                       stretch for readability in rerun (visualization only)
          - luma:  expects [0, 1] brightness, maps directly to [0, 255]
        """
        x = np.asarray(img, dtype=np.float32)
        if mode == "luma":
            x = np.clip(x, 0.0, 1.0)
            return (x * 255.0).astype(np.uint8)
        if mode == "mask":
            x = np.clip(x, 0.0, 1.0)
            return (x * 255.0).astype(np.uint8)
        if mode == "depth_aux":
            # 关键修复：diff_depth 无效像素常为 0。若先全图 clip 到 0.3，
            # 再做分位数拉伸，会被大量无效值主导而整帧接近黑屏。
            # 这里改为：
            # 1) 只在有效深度(>=0.3m)上估计对比度区间
            # 2) 无效像素单独着色为暗灰，既不黑屏也能看出空洞区域
            min_valid = 0.3
            valid = np.isfinite(x) & (x >= min_valid)

            if not np.any(valid):
                # 全无效帧：返回暗灰底图，避免“纯黑=像挂了”的误解
                return np.full_like(x, 18, dtype=np.uint8)

            vals = x[valid]
            if vals.size >= 16:
                lo = float(np.percentile(vals, 2.0))
                hi = float(np.percentile(vals, 98.0))
            else:
                lo = float(vals.min())
                hi = float(vals.max())

            if hi - lo < 1e-4:
                mid = float(np.median(vals))
                lo = max(min_valid, mid - 1.0)
                hi = mid + 1.0

            y = np.zeros_like(x, dtype=np.float32)
            norm = np.clip((x[valid] - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
            # 近处高亮、远处变暗，便于观察障碍轮廓
            y[valid] = 0.25 + 0.75 * np.power(1.0 - norm, 0.8)
            # 无效像素用中灰底色，避免窗口“纯黑像无数据”
            y[~valid] = 0.16
            return (y * 255.0).astype(np.uint8)
        # depth by default
        x = np.clip(x, 0.05, 10.0)
        x = (x - 0.05) / (10.0 - 0.05)
        return (x * 255.0).astype(np.uint8)

    def _log_drone_rig(self, phase: str, pos, drone_R=None, cam_R=None,
                       main_fov_half_tan: float = 0.53,
                       main_hw=(240, 320), depth_hw=(60, 80)):
        """Log drone body size/orientation and a single diff-depth camera frustum in 3D."""
        if not self.enabled:
            return
        rr = self._rr

        o = np.asarray(pos, dtype=np.float32).reshape(3)
        Rw = np.eye(3, dtype=np.float32)
        Rc = np.eye(3, dtype=np.float32)
        if drone_R is not None:
            Rw = np.asarray(drone_R, dtype=np.float32).reshape(3, 3)
        if cam_R is not None:
            Rc = np.asarray(cam_R, dtype=np.float32).reshape(3, 3)

        # Body pose (world)
        rr.log(
            f"{phase}/drone/body",
            rr.Transform3D(translation=o.tolist(), mat3x3=Rw.tolist(), axis_length=0.08),
        )

        # Cinematic body model: chassis + cross arms (local/body coordinates)
        rr.log(
            f"{phase}/drone/body/chassis",
            rr.Boxes3D(
                centers=[[0.0, 0.0, 0.0]],
                sizes=[[0.34, 0.18, 0.08]],
                colors=[[0, 230, 170, 210]],
                radii=[0.01],
                labels=["DRONE BODY"],
                show_labels=False,
            ),
        )
        rr.log(
            f"{phase}/drone/body/arms",
            rr.LineStrips3D(
                [
                    [[-0.20, -0.20, 0.0], [0.20, 0.20, 0.0]],
                    [[-0.20, 0.20, 0.0], [0.20, -0.20, 0.0]],
                ],
                colors=[[100, 255, 255], [100, 255, 255]],
                radii=[0.014, 0.014],
            ),
        )
        # Camera rig
        Rcw = Rw @ Rc

        def _frustum_lines_local(tan_half_x, h, w, near, far):
            tan_half_y = float(tan_half_x) * float(h) / float(max(w, 1))
            c_near = np.array([
                [near, -near * tan_half_x, -near * tan_half_y],
                [near, near * tan_half_x, -near * tan_half_y],
                [near, near * tan_half_x, near * tan_half_y],
                [near, -near * tan_half_x, near * tan_half_y],
            ], dtype=np.float32)
            c_far = np.array([
                [far, -far * tan_half_x, -far * tan_half_y],
                [far, far * tan_half_x, -far * tan_half_y],
                [far, far * tan_half_x, far * tan_half_y],
                [far, -far * tan_half_x, far * tan_half_y],
            ], dtype=np.float32)
            strips = []
            for k in range(4):
                strips.append([[0.0, 0.0, 0.0], c_far[k].tolist()])
                strips.append([c_near[k].tolist(), c_far[k].tolist()])
            strips.append([c_near[0].tolist(), c_near[1].tolist(), c_near[2].tolist(), c_near[3].tolist(), c_near[0].tolist()])
            strips.append([c_far[0].tolist(), c_far[1].tolist(), c_far[2].tolist(), c_far[3].tolist(), c_far[0].tolist()])
            return strips

        # Diff-depth-only branch: show one camera frustum to avoid duplicated rigs.
        rr.log(
            f"{phase}/drone/camera",
            rr.Transform3D(translation=o.tolist(), mat3x3=Rcw.tolist(), axis_length=0.05),
        )
        depth_strips = _frustum_lines_local(float(main_fov_half_tan), int(depth_hw[0]), int(depth_hw[1]), near=0.10, far=0.90)
        rr.log(
            f"{phase}/drone/camera/frustum",
            rr.LineStrips3D(depth_strips, colors=[[80, 200, 255]] * len(depth_strips), radii=[0.0026] * len(depth_strips)),
        )
        rr.log(
            f"{phase}/drone/camera/look",
            rr.Arrows3D(
                origins=[[0.0, 0.0, 0.0]],
                vectors=[[0.62, 0.0, 0.0]],
                colors=[[80, 210, 255]],
                radii=[0.005],
                show_labels=False,
            ),
        )

    def log_step(self, phase: str, step_idx: int, pos, target,
                 depth=None, cam=None, scalars=None,
                 main_img=None, main_img_mode: str = "depth",
                 depth_img=None,
                 quality_img=None,
                 invalid_img=None,
                 scene_effect_img=None,
                 drone_R=None,
                 cam_R=None,
                 main_fov_half_tan: float = 0.53,
                 main_hw=(240, 320),
                 depth_hw=(60, 80)):
        if not self.enabled or self._rr is None:
            return
        rr = self._rr
        try:
            rr.set_time_sequence("step", int(step_idx))
        except Exception as e:
            print(f"[rerun warn] failed to set step time sequence: {e}")
            return

        p = np.asarray(pos, dtype=np.float32).reshape(3)
        t = np.asarray(target, dtype=np.float32).reshape(3)

        phase_path = self._paths.setdefault(phase, [])
        phase_path.append(p.tolist())

        rr.log(
            f"{phase}/drone/pos",
            rr.Points3D([p], colors=[[0, 255, 120]], radii=[0.08], labels=["DRONE"], show_labels=False),
        )
        rr.log(
            f"{phase}/target/pos",
            rr.Points3D([t], colors=[[255, 64, 64]], radii=[0.10], labels=["TARGET"], show_labels=True),
        )
        rr.log(
            f"{phase}/drone/path",
            rr.LineStrips3D([phase_path], colors=[[0, 240, 255]], radii=[0.014]),
        )
        rr.log(
            f"{phase}/drone/path_points",
            rr.Points3D(
                phase_path,
                colors=[[180, 255, 255]] * len(phase_path),
                radii=[0.012] * len(phase_path),
            ),
        )
        rr.log(f"{phase}/metrics/pos_x_m", self._scalar_msg(float(p[0])))
        rr.log(f"{phase}/metrics/pos_y_m", self._scalar_msg(float(p[1])))
        rr.log(f"{phase}/metrics/pos_z_m", self._scalar_msg(float(p[2])))

        # Render drone physical size/orientation + camera rig pose/frustums
        self._log_drone_rig(
            phase=phase,
            pos=p,
            drone_R=drone_R,
            cam_R=cam_R,
            main_fov_half_tan=main_fov_half_tan,
            main_hw=main_hw,
            depth_hw=depth_hw,
        )

        # 新接口：分别记录主相机和深度相机
        if main_img is not None:
            rr.log(f"{phase}/camera/main", rr.Image(self._img_u8(main_img, mode=main_img_mode)))
        # 深度窗口统一为 depth_aux（近亮远暗 + 无效深度暗灰），避免双映射语义冲突。
        depth_for_view = depth_img
        if depth_for_view is None and (main_img is None) and (depth is not None):
            # 兼容旧接口：仅传 depth 时，也统一写入 depth_aux 路径。
            depth_for_view = depth
        if depth_for_view is not None:
            rr.log(f"{phase}/camera/depth_aux", rr.Image(self._img_u8(depth_for_view, mode="depth_aux")))
        if quality_img is not None:
            rr.log(f"{phase}/camera/quality", rr.Image(self._img_u8(quality_img, mode="mask")))
        if invalid_img is not None:
            rr.log(f"{phase}/camera/invalid", rr.Image(self._img_u8(invalid_img, mode="mask")))
        if scene_effect_img is not None:
            rr.log(f"{phase}/camera/scene_effect", rr.Image(self._img_u8(scene_effect_img, mode="mask")))

        if cam is not None:
            power, exposure, gain = [float(x) for x in cam]
            rr.log(f"{phase}/camera/power", self._scalar_msg(power))
            rr.log(f"{phase}/camera/exposure", self._scalar_msg(exposure))
            rr.log(f"{phase}/camera/gain", self._scalar_msg(gain))

        if scalars is not None:
            for k, v in scalars.items():
                rr.log(f"{phase}/metrics/{k}", self._scalar_msg(float(v)))

    def log_train_scalars(self, scalars: dict, iter_idx=None):
        if not self.enabled or self._rr is None:
            return
        rr = self._rr
        # 避免继承 student/teacher 的 step 时间轴。
        # 否则 train 标量会全部堆在同一个 step（例如 70）上，看起来像"没有曲线"。
        try:
            rr.disable_timeline("step")
        except Exception:
            pass
        if iter_idx is not None:
            try:
                rr.set_time_sequence("iter", int(iter_idx))
            except Exception:
                pass
        for k, v in scalars.items():
            try:
                rr.log(f"train/{k}", self._scalar_msg(float(v)))
            except Exception:
                pass
