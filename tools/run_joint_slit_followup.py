#!/usr/bin/env python3
"""Conditional warm start and matched joint-learning arms; no search loops."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from tools.run_joint_slit_learning import (CONFIG, LIGHTS, sha, write_json, atomic_save,
    rng_state, restore_rng, lock_protocol, gate, evaluate, initialize, train_update)
from tools.intent_common import load_intent_args
from config import set_global_seed
from main_cuda import _parameter_hash
from train_utils import build_env
from rollout_ops import (render_gray_sensor, build_local_frame, build_state_vector,
    compute_target_velocity, init_camera_params, update_camera_params)
from controllers.auto_exposure import mean_auto_exposure_target

ARMS = ('A_mean_ae', 'B_full', 'C_no_motion_image', 'D_no_eg_image')


def arm_args(arm):
    if arm not in ARMS:
        raise ValueError(arm)
    return load_intent_args(CONFIG, camera_control_mode='mean_ae' if arm == ARMS[0] else 'learned',
        train_flight_only=arm == ARMS[0], sensor_grad_mode='detached' if arm in (ARMS[0], ARMS[3]) else 'full',
        observation_grad_mode='detached' if arm == ARMS[2] else 'full',
        num_iters=1000, lr=1e-5, camera_lr_multiplier=10.)


def stage_guard(out, directory, extra):
    directory.mkdir(parents=True, exist_ok=True)
    value = dict(protocol_sha256=sha(out/'protocol.json'), followup_sha256=sha(__file__),
                 teacher_source_sha256=sha(ROOT/'tools/audit_slit_information_deadline.py'), **extra)
    path = directory/'inputs.json'
    if path.exists() and json.loads(path.read_text()) != value:
        raise RuntimeError(f'Stage inputs changed: {directory}')
    if not path.exists():
        write_json(path, value)
        (directory/'followup_source.py').write_bytes(Path(__file__).read_bytes())


def save_state(path, model, optimizer, scheduler, iteration, seconds, out):
    atomic_save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
        rng=rng_state(), iteration=iteration, optimization_seconds=seconds, protocol_sha256=sha(out/'protocol.json')), path)


def restore_state(path, model, optimizer, scheduler, out):
    if not path.exists():
        return 0, 0.
    saved = torch.load(path, map_location='cuda')
    if saved['protocol_sha256'] != sha(out/'protocol.json'):
        raise RuntimeError('Resume protocol mismatch')
    model.load_state_dict(saved['model'], strict=True)
    optimizer.load_state_dict(saved['optimizer'])
    scheduler.load_state_dict(saved['scheduler'])
    restore_rng(saved['rng'])
    return saved['iteration'], saved['optimization_seconds']


def truncate_metrics(path, iteration):
    if path.exists():
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        discarded = [row for row in rows if row['iteration'] > iteration]
        if discarded:
            write_json(path.with_name(path.stem+f'_discarded_{time.time_ns()}.json'),
                       dict(reason='Crash after last atomic optimizer state; work retained for cost accounting', rows=discarded))
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows if row['iteration'] <= iteration))


@torch.no_grad()
def collect_demonstrations(out, protocol):
    from tools.audit_slit_information_deadline import MOTIONS, command_for
    args = load_intent_args(CONFIG, batch_size=1)
    target = out/'fallback'
    path = target/'demonstrations.pt'
    if path.exists():
        stored = torch.load(path)
        if stored['specs'] != protocol['demonstrations']:
            raise RuntimeError('Demonstration sample mismatch')
        return stored
    env = build_env(1, args, torch.device('cuda'))
    packed = {key: [] for key in ('frames', 'state', 'action', 'valid')}
    outcomes = []
    start = time.perf_counter()
    for index, spec in enumerate(protocol['demonstrations']):
        set_global_seed(spec['seed'], True)
        env.simple_slit_center_y_min = -.75 if spec['side'] == -1 else 0.
        env.simple_slit_center_y_max = 0. if spec['side'] == -1 else .75
        env.reset(spec['scenario'])
        exposure, gain = init_camera_params(env, 1, torch.device('cuda'))
        queue = [env.act]*2
        alive, reached, hit = True, False, False
        rows = {key: [] for key in packed}
        for step in range(90):
            set_global_seed((spec['seed']*1000003+3*step+1) % (2**32), True)
            frame, _ = render_gray_sensor(env, exposure, gain, differentiable=False)
            local_frame = build_local_frame(env)
            state = build_state_vector(env, compute_target_velocity(env.p_target-env.p, env), local_frame,
                                       exposure, gain, args.no_odom, False)[0]
            command = command_for(env, env.slit_center_y, MOTIONS[0], step, args)
            local_action = (command[:, None] @ local_frame).squeeze(1)
            for key, value in (('frames', frame.half().cpu()), ('state', state.cpu()),
                               ('action', local_action.cpu()), ('valid', torch.tensor([alive]))):
                rows[key].append(value)
            request = mean_auto_exposure_target(frame, exposure, gain)
            exposure, gain, _ = update_camera_params(request, exposure, gain, env)
            queue.append(command)
            set_global_seed((spec['seed']*1000003+3*step+2) % (2**32), True)
            env.run(queue[step], 1/15, env.p_target-env.p)
            collision = bool((env.transition_clearance() <= .0011).item())
            arrival = bool(((env.p-env.p_target).norm(dim=-1) < .35).item())
            if alive:
                hit, reached = collision, arrival and not collision
                alive = not (collision or arrival)
        for key in packed:
            packed[key].append(torch.stack(rows[key], 1))
        outcomes.append(dict(**spec, success=reached, collision=hit))
        if (index+1) % 16 == 0:
            print(f'[demonstrations] {index+1}/128', flush=True)
    result = dict(data={k: torch.cat(v) for k,v in packed.items()}, specs=protocol['demonstrations'],
                  outcomes=outcomes, collection_seconds=time.perf_counter()-start,
                  privileged_flight_labels=True, camera_labels=False, raw_storage='float16 96x72; preprocessing at model only')
    atomic_save(result, path)
    write_json(target/'demonstration_manifest.json', dict(sha256=sha(path), episodes=128,
        generated_steps=128*90, valid_steps=int(result['data']['valid'].sum()),
        collection_seconds=result['collection_seconds'], privileged_flight_labels=True, camera_labels=False,
        successes=sum(r['success'] for r in outcomes), outcomes=outcomes))
    return result


def fallback(out, protocol):
    if (out/'common_locked.json').exists():
        return
    direct = out/'common_direct_decision.json'
    if not direct.exists():
        raise RuntimeError('The bounded direct attempt must finish before imitation')
    target = out/'fallback'
    source = Path(json.loads(direct.read_text())['final_checkpoint'])
    stage_guard(out, target, dict(checkpoint_sha256=sha(source), stage='single_flight_imitation'))
    stored = collect_demonstrations(out, protocol)
    args = load_intent_args(CONFIG)
    model = initialize(args, source)
    model.freeze_camera_for_flight_only()
    camera_hash = _parameter_hash(model, True)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 800, eta_min=1e-5)
    latest = target/'imitation_resume.pt'
    first, seconds = restore_state(latest, model, optimizer, scheduler, out)
    metrics = target/'imitation_metrics.jsonl'
    truncate_metrics(metrics, first)
    data = stored['data']
    for iteration in range(first, 800):
        start = time.perf_counter()
        # Indexed batch samples make restart and logger consumption irrelevant.
        generator = torch.Generator().manual_seed(42+iteration)
        ids = torch.randint(128, (8,), generator=generator)
        images = data['frames'][ids].cuda().float()
        previous = torch.cat([images[:, :1], images[:, :-1]], 1)
        pairs = torch.cat([images, previous], 2)
        state, target_action, valid = (data[k][ids].cuda() for k in ('state', 'action', 'valid'))
        encoded = model.encode_observation(state.flatten(0,1), pairs.flatten(0,1)).reshape(8,90,-1)
        hx, actions = None, []
        for step in range(90):
            action, hx = model.flight_from_encoded(encoded[:, step], hx)
            actions.append(action)
        error = F.smooth_l1_loss(torch.stack(actions,1), target_action, reduction='none').mean(-1)
        loss = (error*valid).sum()/valid.sum().clamp_min(1)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError('Nonfinite imitation loss')
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
        optimizer.step()
        scheduler.step()
        torch.cuda.synchronize()
        seconds += time.perf_counter()-start
        with metrics.open('a') as handle:
            handle.write(json.dumps(dict(iteration=iteration+1, loss=float(loss.detach()),
                valid_steps=int(valid.sum()), cumulative_optimization_seconds=seconds))+'\n')
        if (iteration+1) % 25 == 0:
            save_state(latest, model, optimizer, scheduler, iteration+1, seconds, out)
            print(f'[imitation] {iteration+1}/800 loss={float(loss):.5f}', flush=True)
    if _parameter_hash(model, True) != camera_hash:
        raise RuntimeError('Imitation altered camera-specific weights')
    # Always save/load the same endpoint; no selection of imitation checkpoints.
    atomic_save(model.state_dict(), target/'imitation_final.pth')
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 500, eta_min=3e-7)
    task_latest = target/'task_resume.pt'
    first, seconds = restore_state(task_latest, model, optimizer, scheduler, out)
    env = build_env(4, args, torch.device('cuda'))
    metrics = target/'task_metrics.jsonl'
    truncate_metrics(metrics, first)
    for iteration in range(first, 500):
        row = train_update(model, env, args, optimizer, scheduler, iteration, 110000)
        seconds += row['optimization_seconds']
        row['cumulative_optimization_seconds'] = seconds
        with metrics.open('a') as handle:
            handle.write(json.dumps(row)+'\n')
        if (iteration+1) % 25 == 0:
            save_state(task_latest, model, optimizer, scheduler, iteration+1, seconds, out)
            print(f"[fallback task] {iteration+1}/500 loss={row['loss']:.5f}", flush=True)
    checkpoint = target/'final.pth'
    atomic_save(model.state_dict(), checkpoint)
    normal = evaluate(model, args, protocol['development'], target/'eval_gray.json')
    zero = evaluate(model, args, protocol['development'], target/'eval_zero.json', zero=True)
    decision = dict(**gate(normal, zero), source='single_imitation_warmup', checkpoint=str(checkpoint),
                    checkpoint_sha256=sha(checkpoint), camera_labels=False,
                    camera_specific_unchanged=_parameter_hash(model, True) == camera_hash)
    write_json(target/'decision.json', decision)
    if decision['passed']:
        write_json(out/'common_locked.json', decision)
    else:
        write_json(out/'decision.json', dict(category='common_flight_not_established', evidence=decision,
            next_work='定位基础视觉飞行的学习障碍；不启动联合比较。', complete_experiment_stop=True))


def train_arm(out, protocol, arm):
    common = json.loads((out/'common_locked.json').read_text())
    source = Path(common['checkpoint'])
    if sha(source) != common['checkpoint_sha256']:
        raise RuntimeError('Common checkpoint changed')
    args = arm_args(arm)
    target = out/'arms'/arm
    stage_guard(out, target, dict(stage=arm, args=vars(args), common_sha256=sha(source)))
    if (target/'done.json').exists():
        done = json.loads((target/'done.json').read_text())
        if done['final_sha256'] != sha(target/'final.pth') or done['common_sha256'] != sha(source):
            raise RuntimeError('Completed arm checkpoint mismatch')
        return
    model = initialize(args, source)
    if arm == ARMS[0]:
        model.freeze_camera_for_flight_only()
    groups = []
    for camera, rate in ((False, 1e-5), (True, 1e-4)):
        params = [p for name,p in model.named_parameters() if p.requires_grad and model._is_camera_parameter(name) == camera]
        if params:
            groups.append(dict(params=params, lr=rate))
    optimizer = torch.optim.AdamW(groups)
    # Per-group cosine end factors preserve the 10x ratio throughout.
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda i: .01+.99*(1+__import__('math').cos(__import__('math').pi*min(i,1000)/1000))/2)
    latest = target/'resume.pt'
    first, seconds = restore_state(latest, model, optimizer, scheduler, out)
    if not (target/'initial.pth').exists():
        atomic_save(model.state_dict(), target/'initial.pth')
        write_json(target/'initialization.json', dict(parameters_sha256=_parameter_hash(model),
            total_parameters=sum(p.numel() for p in model.parameters()),
            trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)))
    env = build_env(4, args, torch.device('cuda'))
    metrics = target/'metrics.jsonl'
    truncate_metrics(metrics, first)
    for iteration in range(first, 1000):
        args.camera_handoff_alpha = min(iteration/199, 1.) if arm != ARMS[0] else 1.
        row = train_update(model, env, args, optimizer, scheduler, iteration, 200000)
        seconds += row['optimization_seconds']
        row.update(cumulative_optimization_seconds=seconds, camera_handoff_alpha=args.camera_handoff_alpha)
        with metrics.open('a') as handle:
            handle.write(json.dumps(row)+'\n')
        completed = iteration+1
        if completed % 25 == 0:
            save_state(latest, model, optimizer, scheduler, completed, seconds, out)
            print(f"[{arm}] {completed}/1000 loss={row['loss']:.5f}", flush=True)
        if completed % 200 == 0:
            atomic_save(model.state_dict(), target/f'checkpoint{completed:04d}.pth')
            evaluate(model, args, protocol['development'], target/f'eval_{completed:04d}.json')
    atomic_save(model.state_dict(), target/'final.pth')
    write_json(target/'done.json', dict(updates=1000, generated_steps=360000, optimization_seconds=seconds,
        final_sha256=sha(target/'final.pth'), common_sha256=sha(source), total_parameters=sum(p.numel() for p in model.parameters())))


class CrossPolicy(torch.nn.Module):
    """Offline policy cross: donor camera sees the executed flight's intent."""
    uses_camera_timeline = True

    def __init__(self, flight, camera):
        super().__init__()
        self.flight, self.camera = flight, camera
        self.max_acc_cmd, self.nominal_speed = flight.max_acc_cmd, flight.nominal_speed
        self.camera_hx = None

    def reset(self):
        self.camera_hx = None

    def forward(self, state, hx=None, **kwargs):
        action, _, hx, _ = self.flight(state, hx, **kwargs)
        predictor = kwargs.get('intent_motion_predictor')
        if predictor is None:
            raise RuntimeError('Cross policy requires queue-aligned actual-flight intent')
        camera_kwargs = dict(kwargs, intent_override=predictor(action))
        _, camera, self.camera_hx, _ = self.camera(state, self.camera_hx, **camera_kwargs)
        return action, camera, hx, None


def evaluate_with_frames(model, args, specs, path):
    """Record true sensor frames and executed commands without extra renders."""
    import eval as evaluation
    original = evaluation.run_one_episode
    path = Path(path)
    def recorded(*positional, **kwargs):
        index, env = positional[0], positional[4]
        frames, movements = [], []
        renderer = kwargs.get('sensor_renderer') or render_gray_sensor
        old_run = env.run
        def render(*a, **kw):
            image, auxiliary = renderer(*a, **kw)
            frames.append(image[0,0].detach().cpu().numpy().astype(np.float32))
            return image, auxiliary
        def run(command, dt, target=None):
            pre = dict(velocity=env.v[0].detach().cpu().tolist(),
                       attitude=env.R[0].detach().cpu().tolist(),
                       executed_command=command[0].detach().cpu().tolist())
            result = old_run(command, dt, target)
            pre.update(post_position=env.p[0].detach().cpu().tolist(), dt=float(dt))
            movements.append(pre)
            return result
        env.run = run
        try:
            row, trace = original(*positional, **dict(kwargs, sensor_renderer=render, collect_trace=True))
        finally:
            env.run = old_run
        for step, movement in enumerate(movements):
            trace[step].update(movement)
            trace[step]['exposure_interval_s'] = [trace[step]['time_s']-trace[step]['exposure_us']*1e-6,
                                                   trace[step]['time_s']]
        row['control_cost'] = sum(sum(trace[i][key]**2 for key in
            ('acc_cmd_forward', 'acc_cmd_lateral', 'acc_cmd_vertical'))*movement['dt']
            for i,movement in enumerate(movements))
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path.parent/f'{path.stem}_frames_{index:03d}.npz', frames=np.stack(frames))
        return row, trace
    evaluation.run_one_episode = recorded
    try:
        return evaluate(model, args, specs, path, traces=True)
    finally:
        evaluation.run_one_episode = original


def final_evaluation(out, protocol):
    results = {}
    for arm in ARMS:
        directory = out/'arms'/arm
        done = json.loads((directory/'done.json').read_text())
        checkpoint = directory/'final.pth'
        if sha(checkpoint) != done['final_sha256']:
            raise RuntimeError('Final model changed')
        args = arm_args(arm)
        model = initialize(args, checkpoint)
        stage_guard(out, directory/'confirmation', dict(stage='confirmation', checkpoint_sha256=sha(checkpoint)))
        path = directory/'confirmation/results.json'
        results[arm] = json.loads(path.read_text())['rows'] if path.exists() else evaluate_with_frames(model, args, protocol['confirmation'], path)
    successes = {name: sum(int(r['success_rate']) for r in rows) for name,rows in results.items()}
    crosses = {}
    if successes[ARMS[1]]-successes[ARMS[0]] >= 4:
        stage_guard(out, out/'cross', dict(stage='crosses',
            a_sha256=sha(out/'arms'/ARMS[0]/'final.pth'), b_sha256=sha(out/'arms'/ARMS[1]/'final.pth')))
        a = initialize(arm_args(ARMS[0]), out/'arms'/ARMS[0]/'final.pth')
        b = initialize(arm_args(ARMS[1]), out/'arms'/ARMS[1]/'final.pth')
        for name, model, args in [('B_flight_AE', b, arm_args(ARMS[0])),
                                  ('A_flight_B_camera', CrossPolicy(a,b), arm_args(ARMS[1]))]:
            path = out/'cross'/f'{name}.json'
            rows = json.loads(path.read_text())['rows'] if path.exists() else evaluate_with_frames(model, args, protocol['confirmation'], path)
            crosses[name] = sum(int(r['success_rate']) for r in rows)
    interaction = None
    candidate = False
    if crosses:
        interaction = successes[ARMS[1]]-crosses['B_flight_AE']-crosses['A_flight_B_camera']+successes[ARMS[0]]
        candidate = (successes[ARMS[1]]-successes[ARMS[2]] >= 4 and interaction >= 4
                     and all(successes[ARMS[1]]-value >= 4 for value in crosses.values()))
    write_json(out/'final_results.json', dict(successes=successes, crosses=crosses,
        interaction_success_count=interaction, joint_candidate=candidate, single_training_seed=True,
        caveat='Gradient controls are not RL; crosses may create policy mismatch; no hardware claim.'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['fallback', 'arms', 'evaluate'], required=True)
    parser.add_argument('--arm', choices=ARMS)
    parser.add_argument('--out_dir', type=Path, required=True)
    cli = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    out = cli.out_dir.resolve()
    protocol = lock_protocol(out)
    if cli.stage == 'fallback':
        fallback(out, protocol)
    elif cli.stage == 'arms':
        if not (out/'common_locked.json').exists():
            raise RuntimeError('No accepted common flight checkpoint; arms are forbidden')
        for arm in ((cli.arm,) if cli.arm else ARMS):
            train_arm(out, protocol, arm)
    else:
        final_evaluation(out, protocol)


if __name__ == '__main__':
    main()
