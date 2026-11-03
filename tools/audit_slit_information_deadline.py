#!/usr/bin/env python3
"""Bounded information-time / closed-loop slit diagnostic, never policy training.

Uses the historical 32-element batch to preserve the exact initial state and
disturbance stream. Only the prespecified four layouts per split are scored.
Templates are privileged, noiseless diagnostics; they are not actor inputs.
"""
import argparse
import copy
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from config import set_global_seed
from controllers.auto_exposure import mean_auto_exposure_target, gradient_auto_exposure_target
from model import Model
from render.intent_scene import render_exposure
from render.pose import body_angular_velocity
from rollout_ops import (build_local_frame, decode_action_direct, init_camera_params,
                         update_camera_params, _camera_delay_frames)
from tools.audit_sensing_scale import CONFIG, HISTORICAL, matched_gain, replay_reference
from tools.intent_common import load_intent_args
from train_utils import build_env, write_run_provenance

SPLITS = {'development': (1, 2, 3, 12), 'confirmation': (4, 5, 13, 16)}
SCENARIOS = ('dark', 'bright_to_dark')
MOTIONS = (
    {'id': 0, 'name': 'original', 'x': (-2.5, -.5, .5, 1.5), 'durations': (2.6, 1.4, 1.6)},
    {'id': 1, 'name': 'slow_approach', 'x': (-2.5, -.5, .5, 1.5), 'durations': (3., 1.2, 1.4)},
    {'id': 2, 'name': 'settle_early', 'x': (-2.5, -1., -1., .5, 1.5), 'durations': (1.8, .4, 1.8, 1.6)},
)
RELEASE_STEPS = tuple(range(0, 56, 5))
BIASES_M = (-.04, -.02, -.01, 0., .01, .02, .04)
STEPS, FPS = 90, 15
CLAIM = 'Privileged diagnostic controller/templates; NOT learned navigation or method superiority.'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def preprocess(image):
    return Model.preprocess_gray_input(SimpleNamespace(gray_nn_width=48, gray_nn_height=36),
                                       image, add_noise=False)[:, :1]


def reference(env, estimate, motion, time):
    """No hidden geometry: lateral reference depends only on supplied estimate."""
    durations = motion['durations']
    index = 0
    while index < len(durations) - 1 and time > durations[index]:
        time -= durations[index]
        index += 1
    duration = durations[index]
    u = min(max(time / duration, 0.), 1.)
    z = torch.full_like(estimate, env.simple_slit_center_z)
    points = []
    for k, x in enumerate(motion['x']):
        y = torch.zeros_like(estimate) if k in (0, len(motion['x']) - 1) else estimate
        local = torch.stack([z * 0 + x, y, z], -1)
        points.append((env.R_scene @ local[..., None])[..., 0])
    delta = points[index + 1] - points[index]
    s = 10 * u**3 - 15 * u**4 + 6 * u**5
    ds = (30 * u**2 - 60 * u**3 + 30 * u**4) / duration
    dds = (60 * u - 180 * u**2 + 120 * u**3) / duration**2
    return points[index] + s * delta, ds * delta, dds * delta


def command_for(env, estimate, motion, step, args):
    dt = 1 / FPS
    horizon = 2 * dt + 1 / env.fixed_pitch_ctl_delay
    p, v, a = reference(env, estimate, motion, step * dt + horizon)
    predicted_p = env.p + env.v * horizon + .5 * env.a * horizon**2
    predicted_v = env.v + env.a * horizon
    world = a + 6 * (p - predicted_p) + 4 * (v - predicted_v) + env.fixed_drag_linear * env.v
    frame = build_local_frame(env)
    local = (world[:, None] @ frame).squeeze(1)
    return decode_action_direct(local, frame, env, env.batch_size, args.max_acc_cmd)


class HistoricalBatch:
    def __init__(self, device, verify=False):
        if torch.device(device).type != 'cuda':
            raise ValueError('Exact historical initial-state/RNG replay requires CUDA, as in the original run.')
        self.args = load_intent_args(CONFIG)
        reset_args = copy.deepcopy(self.args)
        reset_args.gray_light_jitter = 0.
        reset_args.gray_enable_noise = False
        set_global_seed(291030)
        self.env = build_env(32, reset_args, torch.device(device))
        self.env.reset('bright_to_dark')
        # Preserve historical factor=1 and all initial state draws; only enable
        # the sensor's existing noise for subsequent image observations.
        self.env.gray_enable_noise = True
        history = json.loads(HISTORICAL.read_text())['reference_dynamics_witness']
        torch.testing.assert_close(self.env.slit_center_y.cpu(),
            torch.tensor(history['slit_centers_y']), rtol=0, atol=1e-6)
        if any(not history['success'][i] for ids in SPLITS.values() for i in ids):
            raise RuntimeError('a locked layout was not historically successful')
        self.devices = [self.env.device.index or 0] if self.env.device.type == 'cuda' else []
        self.rng = []
        self.innovations = []
        with torch.random.fork_rng(devices=self.devices):
            set_global_seed(291031)
            for _ in range(STEPS):
                self.rng.append((torch.get_rng_state().clone(),
                    torch.cuda.get_rng_state(self.env.device).clone() if self.devices else None))
                self.innovations.append(torch.randn_like(self.env.dg))
        self.replay_report = None
        if verify:
            _, _, self.replay_report = replay_reference(self.args, self.env.device)

    def new_env(self, scenario='dark'):
        env = copy.deepcopy(self.env)
        env.current_scene_names = [scenario] * 32
        env.current_scene_name = scenario
        return env

    def run_step(self, env, command, step):
        if env.batch_size != 32:
            raise ValueError('historical disturbance replay requires the original 32-element batch')
        with torch.random.fork_rng(devices=self.devices):
            cpu, cuda = self.rng[step]
            torch.set_rng_state(cpu)
            if self.devices:
                torch.cuda.set_rng_state(cuda, self.env.device)
            env.run(command, 1 / FPS, env.p_target - env.p)

    def noise(self, step, indices):
        # Separate stream, indexed by original layout and frame, independent
        # of method, scenario, template batch size and number of render calls.
        gen = torch.Generator(device=self.env.device).manual_seed(391031 + step)
        shape = (32, 1, 72, 96)
        return tuple(torch.randn(shape, device=self.env.device, generator=gen)[list(indices)]
                     for _ in range(2))


def select_env(env, indices):
    """Slice batch tensors without resetting hardware/scene/physics parameters."""
    out = copy.copy(env)
    ids = torch.as_tensor(indices, device=env.device)
    for name, value in vars(env).items():
        if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == env.batch_size:
            setattr(out, name, value.index_select(0, ids).clone())
    out.batch_size = len(indices)
    out.current_scene_names = [env.current_scene_names[i] for i in indices]
    return out


def set_candidate_centers(env, centers):
    env.voxels[:, :2, 1] += (centers - env.slit_center_y)[:, None]
    env.slit_center_y = centers.clone()


@torch.no_grad()
def observe(history, env, indices, exposure, gain, step):
    sub = select_env(env, indices)
    e, g = exposure[list(indices)], gain[list(indices)]
    ideal, _ = render_exposure(sub, e)
    shot, read = history.noise(step, indices)
    noisy, _ = sub.gray_camera(ideal, e, g, motion=None,
                              shot_noise=shot, read_noise=read, enable_noise=True)
    return preprocess(noisy), noisy


@torch.no_grad()
def template_bank(env, indices, exposure, gain, grid, chunk_size=16):
    """No target label is passed to scoring; overwrite original slit in copies."""
    rows = []
    count = len(indices)
    for start in range(0, len(grid), chunk_size):
        chunk = grid[start:start + chunk_size]
        sub = select_env(env, [i for i in indices for _ in chunk])
        centers = chunk.repeat(count)
        set_candidate_centers(sub, centers)
        e = exposure[list(indices)].repeat_interleave(len(chunk))
        g = gain[list(indices)].repeat_interleave(len(chunk))
        ideal, _ = render_exposure(sub, e)
        clean, _ = sub.gray_camera(ideal, e, g, motion=None, enable_noise=False)
        rows.append(preprocess(clean).reshape(count, len(chunk), 1, 36, 48))
    return torch.cat(rows, 1)


def choose_template(current, templates, previous_estimate, grid, previous=None):
    scores = (templates - current[:, None]).square().flatten(2).sum(-1)
    if previous is not None:
        old_image, old_templates = previous
        scores = scores + (old_templates - old_image[:, None]).square().flatten(2).sum(-1)
    # Exact forward ties only. Nearest previous estimate, then low grid index.
    tied = scores == scores.amin(1, keepdim=True)
    distance = (grid[None] - previous_estimate[:, None]).abs()
    distance = torch.where(tied, distance, torch.full_like(distance, float('inf')))
    return grid[distance.argmin(1)]


def camera_options(calibration):
    result = [dict(id=i, name=f'fixed_e{e:.2f}', kind='fixed', exposure=e, gain=.02,
                   available=True) for i, e in enumerate((.10, .60, .85))]
    for i, e in enumerate((.10, .60), 3):
        match = matched_gain(calibration, e, .85)
        result.append(dict(id=i, name=f'matched_e{e:.2f}', kind='fixed', exposure=e,
                           gain=match.get('gain01'), available=match['available'], match=match))
    result.extend([dict(id=5, name='mean_ae', kind='ae', available=True),
                   dict(id=6, name='gradient_ae', kind='ae', available=True)])
    return result


def release_estimate(center, step, release_step, bias):
    return center + bias if step >= release_step else torch.zeros_like(center)


@torch.no_grad()
def rollout(history, motion, *, mode, indices, scenario='dark', release_step=0,
            bias=0., camera=None, record=True):
    args, env = history.args, history.new_env(scenario)
    n = env.batch_size
    estimate = torch.zeros_like(env.slit_center_y)
    queue = [env.act.clone(), env.act.clone()]
    alive = torch.ones(n, dtype=torch.bool, device=env.device)
    success, collision, crossed = ~alive, ~alive, ~alive
    time = env.p.new_full((n,), 6.)
    minimum = env.p.new_full((n,), 100.)
    effort = env.p.new_zeros(n)
    first_cross = torch.full((n,), -1, device=env.device, dtype=torch.long)
    hit_position = env.p.new_zeros(n, 3)
    env.camera_control_mode = 'learned'  # Diagnostic commands use normal delay/smoothing.
    e, g = init_camera_params(env, n, env.device)
    if mode == 'image' and camera is None:
        raise ValueError('image mode requires a camera option')
    grid = torch.arange(round(args.simple_slit_center_y_min * 100),
                        round(args.simple_slit_center_y_max * 100) + 1, device=env.device) / 100.
    previous = None
    traces, frames = [], []
    ids = list(indices)
    # IDs in timelines refer to requests; -1 is the initialized camera setting.
    delay = _camera_delay_frames(env)
    for step in range(STEPS):
        image, noisy = None, None
        if mode == 'oracle':
            estimate = env.slit_center_y.clone()
        elif mode == 'release':
            estimate = release_estimate(env.slit_center_y, step, release_step, bias)
        elif mode == 'image':
            image, noisy = observe(history, env, ids, e, g, step)
            templates = template_bank(env, ids, e, g, grid)
            estimate[ids] = choose_template(image, templates, estimate[ids], grid, previous)
            previous = (image, templates)
        else:
            raise ValueError(f'unknown mode {mode}')
        command = command_for(env, estimate, motion, step, args)
        executed = queue.pop(0)
        queue.append(command)
        applied = torch.stack([e, g], -1)
        if mode == 'image':
            if camera['name'] == 'mean_ae':
                target = applied.clone()
                target[ids] = mean_auto_exposure_target(noisy, e[ids], g[ids])
            elif camera['name'] == 'gradient_ae':
                target = applied.clone()
                target[ids] = gradient_auto_exposure_target(noisy, e[ids], g[ids])
            else:
                target = e.new_tensor([camera['exposure'], camera['gain']])[None].expand(n, -1)
        else:
            target = applied
        duration = env.gray_camera.calibration.exposure_to_us(e) * 1e-6
        if record:
            local_p = (env.R_scene_T @ env.p[..., None])[..., 0]
            row = dict(step=step, time_s=step / FPS, local_position=local_p[ids].cpu().tolist(),
                position=env.p[ids].cpu().tolist(), velocity=env.v[ids].cpu().tolist(),
                rotation=env.R[ids].cpu().tolist(),
                angular_velocity=body_angular_velocity(env.R_old, env.R, 1 / FPS)[ids].cpu().tolist(),
                estimate_m=estimate[ids].cpu().tolist(),
                error_m=(estimate-env.slit_center_y)[ids].cpu().tolist(),
                submitted_command=command[ids].cpu().tolist(), executed_command=executed[ids].cpu().tolist(),
                applied_eg=applied[ids].cpu().tolist(), requested_eg=target[ids].cpu().tolist(),
                command_id=step, effective_frame=step + delay + 1,
                applied_command_id=step - delay - 1 if step > delay else -1,
                exposure_start_s=(step / FPS - duration[ids]).cpu().tolist(),
                exposure_end_s=step / FPS, alive_before=alive[ids].cpu().tolist())
        e, g, _ = update_camera_params(target, e, g, env)
        effort += alive * command.square().sum(-1) / FPS
        history.run_step(env, executed, step)
        clearance = env.transition_clearance()
        minimum = torch.minimum(minimum, torch.where(alive, clearance, minimum))
        hit = env.collision_since_last_step() | (clearance <= args.collision_clearance)
        goal = (env.p - env.p_target).norm(dim=-1) < .35
        local_x = (env.R_scene_T @ env.p[..., None])[:, 0, 0]
        # Entire spherical body has cleared the rear face, using same wall geometry.
        past = local_x >= args.simple_wall_x + env.simple_wall_half_x + env.drone_radius
        newly_crossed = alive & ~hit & past & ~crossed
        first_cross = torch.where(newly_crossed, first_cross.new_full((n,), step), first_cross)
        crossed |= newly_crossed
        hit_position = torch.where((alive & hit)[:, None], env.p, hit_position)
        collision |= alive & hit
        success |= alive & ~hit & goal
        done = alive & (hit | goal)
        time = torch.where(done, time.new_full((n,), (step + 1) / FPS), time)
        alive &= ~done
        if record:
            row.update(clearance_m=clearance[ids].cpu().tolist(), collision=collision[ids].cpu().tolist(),
                       success=success[ids].cpu().tolist(), crossed=crossed[ids].cpu().tolist())
            traces.append(row)
            if noisy is not None and step % 5 == 0:
                frames.append(dict(step=step, images=noisy[:, 0].cpu().numpy()))
        if mode == 'image' and not alive[ids].any():
            break
    episodes = [dict(episode_index=i, scenario=scenario, success=bool(success[i]),
        collision=bool(collision[i]), timeout=bool(alive[i]), crossed=bool(crossed[i]),
        cross_step=int(first_cross[i]), terminal_time_s=float(time[i]),
        scored_time_s=float(time[i]) if success[i] else 6., control_cost=float(effort[i]),
        min_clearance_m=float(minimum[i]),
        collision_position=hit_position[i].cpu().tolist() if collision[i] else None,
        true_center_m=float(env.slit_center_y[i])) for i in ids]
    return dict(motion=motion['name'], mode=mode, scenario=scenario, release_step=release_step,
                bias_m=bias, camera=camera, episodes=episodes, timeline=traces, claim=CLAIM), frames


def success_count(row):
    return sum(e['success'] for e in row['episodes'])


def rank(row):
    return (-success_count(row), sum(e['scored_time_s'] for e in row['episodes']),
            sum(e['control_cost'] for e in row['episodes']),
            next(m['id'] for m in MOTIONS if m['name'] == row['motion']), row['camera']['id'])


def lock_candidates(rows, eligible):
    """Pure development-only selection; original motion is required for attribution."""
    if 'original' not in eligible:
        return dict(status='control_confounded', reason='Original process failed its 8/8 control reference.', arms=[])
    ae = [r for r in rows if r['camera']['kind'] == 'ae']
    fixed = [r for r in rows if r['camera']['kind'] == 'fixed']
    if not ae or not fixed:
        return dict(status='control_confounded', reason='Required control arms are unavailable.', arms=[])
    # Choose one AE family on development and retain it in all crossed arms.
    # No extra motion-specific AE selection, and at most six unique arms.
    family = min(ae, key=rank)['camera']['name']
    ae = [r for r in ae if r['camera']['name'] == family]
    base_ae = next(r for r in ae if r['motion'] == 'original')
    best_ae = min(ae, key=rank)
    base_fixed = min([r for r in fixed if r['motion'] == 'original'], key=rank)
    joint = min(fixed, key=rank)
    joint_ae = next(r for r in ae if r['motion'] == joint['motion'])
    crossed_fixed = next(r for r in fixed if r['motion']=='original' and r['camera']==joint['camera'])
    crossed_ae = next(r for r in ae if r['motion']=='original' and r['camera']==joint_ae['camera'])
    roles = dict(base_ae=base_ae, best_ae=best_ae, base_fixed=base_fixed, joint=joint,
                 cross_fixed=crossed_fixed, joint_ae=joint_ae, cross_ae=crossed_ae)
    arms, role_ids = [], {}
    for role, row in roles.items():
        arm = dict(motion=row['motion'], camera=row['camera'])
        if arm not in arms:
            arms.append(arm)
        role_ids[role] = arms.index(arm)
    assert len(arms) <= 6
    return dict(status='locked', arms=arms, role_ids=role_ids,
                ae_family=family,
                development_counts={role: success_count(row) for role,row in roles.items()},
                selection_rule='success, failure-penalized time, command effort, motion id, camera id')


def decision(lock, confirmation, oracle_ok):
    if lock['status'] != 'locked' or not all(oracle_ok.values()):
        return dict(category='control_confounded', label='控制参考未通过，无法判断联合收益')
    counts = {role: success_count(confirmation[index]) for role,index in lock['role_ids'].items()}
    dev = lock['development_counts']
    interaction = counts['joint'] - counts['joint_ae'] - counts['cross_fixed'] + counts['cross_ae']
    dev_interaction = dev['joint'] - dev['joint_ae'] - dev['cross_fixed'] + dev['cross_ae']
    candidate = (counts['joint'] - counts['best_ae'] >= 2 and
                 counts['joint'] - counts['base_fixed'] >= 2 and interaction >= 1 and
                 dev['joint'] > dev['best_ae'] and dev['joint'] > dev['base_fixed'] and dev_interaction > 0)
    if candidate:
        category, label = 'joint_candidate', '出现联合任务收益候选；下一轮只检查对应学习信号'
    elif all(success_count(r) >= 7 for r in confirmation):
        category, label = 'ceiling', '各方案几乎全部成功，当前检查缺乏区分能力'
    else:
        category, label = 'no_joint_value', '当前测试未支持额外联合价值，停止扩展'
    return dict(category=category, label=label, counts=counts, interaction_successes=interaction,
                development_interaction_successes=dev_interaction)


def summarize_windows(rows):
    windows = []
    for motion in MOTIONS:
        group = [r for r in rows if r['motion'] == motion['name']]
        if not group:
            continue
        for i in (*SPLITS['development'], *SPLITS['confirmation']):
            valid = {}
            for row in group:
                e = next(e for e in row['episodes'] if e['episode_index'] == i)
                if e['success']:
                    valid.setdefault(str(row['release_step']), []).append(row['bias_m'])
            # Preserve nonmonotonic/disjoint success sets. No inferred boundary.
            zero_steps = [r['release_step'] for r in group if r['bias_m']==0. and
                          next(e for e in r['episodes'] if e['episode_index']==i)['success']]
            windows.append(dict(motion=motion['name'], episode_index=i,
                successful_biases_by_release_step=valid, zero_bias_successful_steps=zero_steps,
                last_sampled_zero_bias_success_step=max(zero_steps) if zero_steps else None))
    return windows


def usable_information(row, information_rows):
    """Post-hoc error/time descriptors, never a control input or inferred bound."""
    results=[]
    for i,episode in enumerate(row['episodes']):
        relevant=[r for r in information_rows if r['motion']==row['motion']]
        close=episode['cross_step']
        if close < 0:
            results.append(dict(episode_index=episode['episode_index'],first_stable_step=None,
                                reason='did_not_safely_clear_wall'));continue
        errors=[abs(t['error_m'][i]) for t in row['timeline'][:close+1]]
        # A practical precision descriptor, not an exact mapping from varying
        # online error histories to one-time constant-bias release experiments.
        stable={str(b):next((s for s in range(len(errors)) if max(errors[s:]) <= b),None)
                for b in (.005,.01,.02,.04)}
        zero_steps=[r['release_step'] for r in relevant if r['bias_m']==0 and
            next(e for e in r['episodes'] if e['episode_index']==episode['episode_index'])['success']]
        results.append(dict(episode_index=episode['episode_index'],cross_step=close,
            stable_error_threshold_steps=stable,zero_bias_successful_release_steps=zero_steps,
            caution='Varying image errors differ from a single constant-bias release; thresholds are descriptive.'))
    return results


def protocol(history):
    return dict(claim=CLAIM, policy_training=False, calibrated=False, splits=SPLITS,
        scenarios=SCENARIOS, motions=MOTIONS, release_steps=RELEASE_STEPS, biases_m=BIASES_M,
        fps=FPS, deadline_s=6, sensor_wh=[96,72], input_wh=[48,36], grid_step_m=.01,
        reset_seed=291030, dynamics_seed=291031, noise_stream='391031 + frame step; 32-layout indexed',
        historical_initial_light_factor=1.,
        sample_scope='Historical tracker-successful layouts, NOT independent new held-out generalization.',
        cameras=camera_options(history.env.gray_camera.calibration),
        control_gate='8/8 development; original process required for crossed attribution',
        geometry_lateral_margin_m=history.args.simple_slit_half_y-.12-history.args.collision_clearance,
        early_stop='No eligible original control => no camera comparison, no training, no retuning.',
        source_sha256={name:sha(ROOT/name) for name in (
            'tools/audit_slit_information_deadline.py','tools/audit_intent_scene.py','model.py',
            'rollout_ops.py','env_cuda.py','render/intent_scene.py','render/slit_scene.py',
            'controllers/auto_exposure.py',CONFIG)},
        calibration_sha256=sha(ROOT/history.args.imx900_calibration),
        historical_sha256=sha(HISTORICAL),
        git_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=ROOT).strip())


def require_protocol(out):
    p = json.loads((out/'protocol.json').read_text())
    for name, expected in p['source_sha256'].items():
        if sha(ROOT/name) != expected:
            raise RuntimeError(f'protocol source changed: {name}; do not reuse this run')
    return p


def information_stage(out, history):
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/'protocol.json', protocol(history))
    write_run_provenance(out, history.args)
    if history.replay_report:
        write_json(out/'historical_replay.json', history.replay_report)
    np.savez_compressed(out/'historical_initial_state.npz',
        **{k:v.cpu().numpy() for k,v in history.env.save_state().items()},
        centers=history.env.slit_center_y.cpu().numpy(),
        innovations=torch.stack(history.innovations).cpu().numpy())
    ids = (*SPLITS['development'], *SPLITS['confirmation'])
    controls, rows = [], []
    for motion in MOTIONS:
        row, _ = rollout(history, motion, mode='oracle', indices=ids)
        controls.append(row)
        write_json(out/f'oracle_{motion["name"]}.json', row)
        for split, selected in SPLITS.items():
            print('CONTROL',motion['name'],split,
                sum(e['success'] for e in row['episodes'] if e['episode_index'] in selected)*2,'/8',flush=True)
        for release in RELEASE_STEPS:
            for bias in BIASES_M:
                row, _ = rollout(history, motion, mode='release', indices=ids,
                                 release_step=release, bias=bias, record=False)
                rows.append(row)
            print('INFORMATION',motion['name'],'release',release,flush=True)
        write_json(out/'information_rows.json',rows)
    write_json(out/'information_windows.json', summarize_windows(rows))
    gates = {split:{r['motion']:all(e['success'] for e in r['episodes']
                                  if e['episode_index'] in selected)
                    for r in controls} for split,selected in SPLITS.items()}
    write_json(out/'control_gates.json',gates)


def development_stage(out, history):
    require_protocol(out)
    if (out/'locked_candidates.json').exists():
        raise RuntimeError('development is already locked; do not reselect')
    gates = json.loads((out/'control_gates.json').read_text())['development']
    eligible = [name for name, ok in gates.items() if ok]
    rows = []
    # Full factorial cannot support the prespecified contrasts without the
    # original process; stop early rather than select a different comparison.
    if 'original' in eligible:
        for motion in MOTIONS:
            if motion['name'] not in eligible:
                continue
            for camera in camera_options(history.env.gray_camera.calibration):
                if not camera['available']:
                    continue
                combined = dict(motion=motion['name'],camera=camera,episodes=[])
                for scenario in SCENARIOS:
                    row, frames = rollout(history,motion,mode='image',indices=SPLITS['development'],
                                          scenario=scenario,camera=camera)
                    write_json(out/f'dev_{motion["name"]}_{camera["name"]}_{scenario}.json',row)
                    np.savez_compressed(out/f'dev_{motion["name"]}_{camera["name"]}_{scenario}_frames.npz',
                                        steps=[f['step'] for f in frames],images=[f['images'] for f in frames])
                    combined['episodes'].extend(row['episodes'])
                rows.append(combined)
                write_json(out/'development_rows.json', rows)
                print('DEVELOPMENT',motion['name'],camera['name'],success_count(combined),'/8',flush=True)
    write_json(out/'development_rows.json',rows)
    write_json(out/'locked_candidates.json',lock_candidates(rows,eligible))


def confirmation_stage(out, history):
    require_protocol(out)
    if (out/'confirmation_rows.json').exists():
        raise RuntimeError('confirmation already completed; no repeat/reselection')
    lock = json.loads((out/'locked_candidates.json').read_text())
    lock_hash = sha(out/'locked_candidates.json')
    rows = []
    for index, arm in enumerate(lock['arms']):
        motion = next(m for m in MOTIONS if m['name']==arm['motion'])
        combined = dict(motion=motion['name'],camera=arm['camera'],episodes=[])
        gates=json.loads((out/'control_gates.json').read_text())['confirmation']
        if not gates[motion['name']]:
            write_json(out/'confirmation_rows.json',[])
            write_json(out/'confirmation_lock.json',dict(locked_candidates_sha256=lock_hash,
                stopped='Selected motion failed confirmation control reference.'))
            return
        for scenario in SCENARIOS:
            row, frames = rollout(history,motion,mode='image',indices=SPLITS['confirmation'],
                scenario=scenario,camera=arm['camera'])
            write_json(out/f'confirm_{index}_{scenario}.json',row)
            np.savez_compressed(out/f'confirm_{index}_{scenario}_frames.npz',
                steps=[f['step'] for f in frames],images=[f['images'] for f in frames])
            combined['episodes'].extend(row['episodes'])
        rows.append(combined)
        print('CONFIRMATION',index,success_count(combined),'/8',flush=True)
    if sha(out/'locked_candidates.json') != lock_hash:
        raise RuntimeError('locked candidates changed during confirmation')
    write_json(out/'confirmation_rows.json',rows)
    write_json(out/'confirmation_lock.json',dict(locked_candidates_sha256=lock_hash))


def make_figure(out, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig, axes = plt.subplots(2,3,figsize=(14,8),layout='constrained')
    rows = json.loads((out/'information_rows.json').read_text())
    for col,motion in enumerate(MOTIONS):
        ax=axes[0,col]
        counts=np.array([[sum(e['success'] for e in r['episodes']) for release in RELEASE_STEPS
                         for r in rows if r['motion']==motion['name'] and r['release_step']==release
                         and r['bias_m']==bias] for bias in BIASES_M])
        im=ax.imshow(counts,origin='lower',aspect='auto',vmin=0,vmax=8,cmap='Blues')
        ax.set(title=motion['name']+' | successes / 8 historical layouts',
               xlabel='Information release time (s)',ylabel='Position bias (cm)')
        ax.set_xticks(range(0,len(RELEASE_STEPS),2),[f'{s/FPS:.2f}' for s in RELEASE_STEPS[::2]])
        ax.set_yticks(range(len(BIASES_M)),[f'{b*100:g}' for b in BIASES_M])
    fig.colorbar(im,ax=list(axes[0]),label='Diagnostic arrivals / 8')
    # Full-information original trajectory; chosen index fixed before observing results.
    control=json.loads((out/'oracle_original.json').read_text())
    ids=(*SPLITS['development'],*SPLITS['confirmation'])
    i=0
    trace=control['timeline']; center=control['episodes'][i]['true_center_m']
    axes[1,0].plot([t['local_position'][i][0] for t in trace],[t['local_position'][i][1] for t in trace])
    axes[1,0].axvline(0,color='grey');axes[1,0].plot([0,0],[center-.15,center+.15],color='green',lw=4)
    axes[1,0].set(title=f'Known-center reference, historical index {ids[i]}',xlabel='x (m)',ylabel='y (m)')
    lock=json.loads((out/'locked_candidates.json').read_text())
    if summary['camera_comparison_executed']:
        j=lock['role_ids']['joint']
        frame_path=out/f'confirm_{j}_dark_frames.npz'
        f=np.load(frame_path);k=min(6,len(f['steps'])-1)
        axes[1,1].imshow(f['images'][k,0],cmap='gray',vmin=0,vmax=1)
        axes[1,1].set_title(f'Joint candidate, dark, t={f["steps"][k]/FPS:.2f}s');axes[1,1].axis('off')
        ax=axes[1,2]
        for role,color in (('base_ae','#3b729a'),('joint','#ad6a34')):
            idx=lock['role_ids'][role]
            r=json.loads((out/f'confirm_{idx}_dark.json').read_text())
            live=[t for t in r['timeline'] if t['alive_before'][0]]
            ax.plot([t['time_s'] for t in live],[t['error_m'][0]*100 for t in live],label=role,color=color)
        ax.axhline(0,color='gray',lw=.6);ax.set(xlabel='Time (s)',ylabel='Estimated center error (cm)',title='Historical index 4; diagnostic only')
        ax.legend(frameon=False)
    else:
        axes[1,1].axis('off');axes[1,1].text(.1,.5,'Image comparison not run:\ncontrol reference failed',transform=axes[1,1].transAxes)
        axes[1,2].axis('off')
        axes[1,2].text(.02,.95,'Decision: '+summary['decision']['category']+'\n\n'+
            'No policy training; no retuning.\nHistorical successful layouts only.\n'+
            'All motion references use the same 6 s deadline.\n\n'+
            'Image estimation error: unavailable\nwhen control gate prevents image comparison.',
            va='top',wrap=True,transform=axes[1,2].transAxes)
    fig.suptitle('When must the drone know the slit position?\nProvisional simulator; diagnostic controller only',fontsize=14)
    fig.savefig(out/'comparison.png',dpi=150)
    fig.savefig(out/'comparison_preview.jpg',dpi=65,pil_kwargs={'quality':78})
    plt.close(fig)


def summary_stage(out):
    p=require_protocol(out)
    lock=json.loads((out/'locked_candidates.json').read_text())
    rows=json.loads((out/'confirmation_rows.json').read_text())
    gates=json.loads((out/'control_gates.json').read_text())
    required={a['motion']:gates['confirmation'][a['motion']] for a in lock['arms']}
    result=dict(claim=CLAIM,calibrated=False,policy_training=False,control_gates=gates,
        decision=decision(lock,rows,required),confirmation=rows,
        camera_comparison_executed=bool(rows),information_rollouts=3*len(RELEASE_STEPS)*len(BIASES_M),
        information_examples='8 unique historical layouts; lighting duplicated only in camera stage',
        limitations=['No theoretical reachability bound. No learned-policy conclusion.',
            'Conditional on prespecified historical successful layouts; not independent generalization.',
            'Historical light factor=1 and original disturbance stream preserved.',
            'Image methods are skipped when original control reference fails.'],protocol_sha256=sha(out/'protocol.json'))
    write_json(out/'summary.json',result)
    if rows:
        info=json.loads((out/'information_rows.json').read_text())
        descriptors={}
        for i in range(len(rows)):
            for scenario in SCENARIOS:
                r=json.loads((out/f'confirm_{i}_{scenario}.json').read_text())
                descriptors[f'{i}_{scenario}']=usable_information(r,info)
        write_json(out/'image_information_timing.json',descriptors)
    make_figure(out,result)
    write_json(out/'artifact_hashes.json',{f.name:sha(f) for f in sorted(out.iterdir())
        if f.is_file() and f.name!='artifact_hashes.json'})
    print('DECISION',result['decision'],flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out_dir',required=True)
    parser.add_argument('--device',default='cuda',choices=('cuda',))
    parser.add_argument('--stage',default='all',choices=('all','information','development','confirmation','summary'))
    cli=parser.parse_args(); torch.set_num_threads(4)
    out=Path(cli.out_dir).resolve()
    if cli.stage=='summary':
        summary_stage(out);return
    history=HistoricalBatch(cli.device,verify=cli.stage in ('all','information'))
    if cli.stage in ('all','information'):
        information_stage(out,history)
    if cli.stage in ('all','development'):
        development_stage(out,history)
    if cli.stage in ('all','confirmation'):
        confirmation_stage(out,history)
    if cli.stage=='all':
        summary_stage(out)


if __name__=='__main__':
    main()
