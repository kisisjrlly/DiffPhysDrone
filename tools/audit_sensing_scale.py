#!/usr/bin/env python3
"""Bounded same-frame resolution diagnostic; no navigation/camera policy training.

Replays the original privileged slit tracking witness, selects its first
successful episode without looking at images, and fits only the existing
fixed-rule ridge image probe. Sensor resolution/radiometry remain unchanged.
"""
import argparse
import copy
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from config import set_global_seed
from model import Model
from render.intent_scene import render_exposure, render_scene_irradiance
from render.pose import body_angular_velocity, rotation_exp
from tools.audit_intent_scene import executed_dynamics_audit
from tools.audit_slit_precision import ridge_probe
from tools.intent_common import load_intent_args
from train_utils import build_env, write_run_provenance

CONFIG = 'configs/gray_joint_slit.args'
HISTORICAL = ROOT / 'docs/grayscale_imx900/joint_slit_audit_20261029/scene_audit.json'
EXPOSURES = (.10, .60, .85)
RESOLUTIONS = ((48, 36), (96, 72))  # width, height
SCENARIOS = ('dark', 'bright_to_dark')
DISTANCES = (-1., -.4)
COUNT = 32
CENTER_DIFFERENCE_M = .005
THRESHOLD = .10


def digest_tensor(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def resolution_inputs(noisy):
    """Both branches consume the SAME noisy sensor tensor, using policy preprocessing.

    One channel is sufficient for this single-frame probe. The policy's duplicated
    second channel carries no extra information here; no temporal policy is fitted.
    """
    if tuple(noisy.shape[1:]) != (1, 72, 96):
        raise ValueError('this bounded diagnostic requires native 96x72 grayscale')
    return {f'{w}x{h}': Model.preprocess_gray_input(
        SimpleNamespace(gray_nn_width=w, gray_nn_height=h), noisy, add_noise=False
    )[:, :1] for w, h in RESOLUTIONS}


def sensor_branches(camera, irradiance, exposure, gain, shot, read):
    """Noise is applied once upstream of the two resolution branches."""
    noisy, aux = camera(irradiance, exposure, gain, motion=None, shot_noise=shot,
                        read_noise=read, enable_noise=True)
    return noisy, resolution_inputs(noisy), aux


@torch.no_grad()
def replay_reference(args, device):
    """Same 32 historical layouts/RNG streams; no new episode/seed screening."""
    replay_args = copy.deepcopy(args)
    replay_args.batch_size = 32
    replay_args.gray_light_jitter = 0.
    replay_args.gray_enable_noise = False
    set_global_seed(291030)
    env = build_env(32, replay_args, device)
    env.reset('bright_to_dark')
    center = env.slit_center_y
    z = torch.full_like(center, env.simple_slit_center_z)
    points = torch.stack([
        torch.stack([z * 0 + args.simple_start_x, z * 0, z], -1),
        torch.stack([z * 0 - .5, center, z], -1),
        torch.stack([z * 0 + .5, center, z], -1),
        torch.stack([z * 0 + args.simple_goal_x, z * 0, z], -1),
    ], 1)
    set_global_seed(291031)
    witness = executed_dynamics_audit(env, replay_args, local_points=points,
        durations=[2.6, 1.4, 1.6], record_camera_states=True)
    historical = json.loads(HISTORICAL.read_text())['reference_dynamics_witness']
    if witness['success'] != historical['success']:
        raise RuntimeError('original success mask did not reproduce; do not select a new trajectory')
    torch.testing.assert_close(center.cpu(), torch.tensor(historical['slit_centers_y']), rtol=0, atol=1e-6)
    raw_path = ROOT / 'logs/joint_slit_20261029/scene_audit_v2/scene_audit.json'
    difference = None
    if raw_path.exists():
        raw = json.loads(raw_path.read_text())['reference_dynamics_witness']
        difference = float((torch.tensor(witness['local_trajectory']) -
                            torch.tensor(raw['local_trajectory'])).abs().max())
        if difference > 1e-5:
            raise RuntimeError(f'original trajectory differs by {difference}; stop before image screen')
    for key in ('collision', 'terminal_time_s', 'min_transition_clearance_m'):
        torch.testing.assert_close(torch.tensor(witness[key]), torch.tensor(historical[key]), rtol=0, atol=1e-5)
    episode = witness['success'].index(True)
    snapshots = []
    for requested_x in DISTANCES:
        candidates = []
        for step, record in enumerate(witness['camera_states']):
            p = torch.tensor(record['p'][episode], device=device)
            local_p = env.R_scene_T[episode] @ p
            if record['alive'][episode] and float(local_p[0]) < args.simple_wall_x:
                candidates.append((abs(float(local_p[0]) - requested_x), step, record))
        _, step, record = min(candidates, key=lambda item: (item[0], item[1]))
        state = {key: torch.tensor(record[key][episode], device=device) for key in ('p', 'v', 'R', 'R_old')}
        state.update(requested_x=requested_x, step=step, time_s=record['time_s'],
                     center=float(center[episode]), episode=episode)
        snapshots.append(state)
    report = {key: value for key, value in witness.items() if key != 'camera_states'}
    report['selected_states'] = [{key: value.detach().cpu().tolist() if isinstance(value, torch.Tensor) else value
                                  for key,value in state.items()} for state in snapshots]
    report.update(selected_episode=episode, selection='lowest index successful episode; nearest active x before wall',
                  historical_trajectory_max_abs_error=difference, reset_seed=291030, dynamics_seed=291031,
                  historical_result_sha256=hashlib.sha256(HISTORICAL.read_bytes()).hexdigest())
    return env, snapshots, report


def nuisance(seed, count, device):
    generator = torch.Generator(device=device).manual_seed(seed)
    return dict(yaw=(torch.rand(count, device=device, generator=generator) * 2 - 1) * math.radians(.25),
                phase=torch.rand(count, device=device, generator=generator) * 2 * math.pi,
                light=.92 + .16 * torch.rand(count, device=device, generator=generator),
                shot=torch.randn(count, 1, 72, 96, device=device, generator=generator),
                read=torch.randn(count, 1, 72, 96, device=device, generator=generator))


def image_environment(args, reference, state, scenario, moving, center_class, draws, count):
    env = build_env(count, args, reference.device)
    env.reset(scenario)
    i = state['episode']
    env.voxels = reference.voxels[i:i+1].expand(count, -1, -1).clone()
    for key in ('R_scene', 'R_scene_T', 'R_cam'):
        value = getattr(reference, key)
        setattr(env, key, value[i:i+1].expand(count, -1, -1).clone())
    center = state['center'] + center_class * CENTER_DIFFERENCE_M / 2
    env.voxels[:, :2, 1] += center - state['center']
    env.slit_center_y = env.voxels.new_full((count,), center)
    env.p = state['p'][None].expand(count, -1).clone()
    env.v = state['v'][None].expand(count, -1).clone() if moving else torch.zeros_like(env.p)
    residual = torch.zeros(count, 3, device=reference.device)
    residual[:, 2] = draws['yaw']
    env.R = state['R'][None] @ rotation_exp(residual)
    dt = 1 / args.base_control_freq
    omega = body_angular_velocity(state['R_old'][None], state['R'][None], dt)
    env.R_old = env.R @ rotation_exp(-omega.expand(count, -1) * dt) if moving else env.R.clone()
    env.last_control_dt = dt
    env.gray_texture_phase = draws['phase']
    env.gray_light_factor = draws['light']
    return env


@torch.no_grad()
def samples(args, reference, state, scenario, moving, exposure, gain, split, center_class):
    seed = (391000 if split == 'fit' else 491000) + (center_class + 1) * 100
    draws = nuisance(seed, COUNT, reference.device)
    env = image_environment(args, reference, state, scenario, moving, center_class, draws, COUNT)
    e = env.p.new_full((COUNT,), exposure)
    g = env.p.new_full((COUNT,), gain)
    irradiance, optical = render_exposure(env, e, return_aux=True)
    noisy, branches, aux = sensor_branches(env.gray_camera, irradiance, e, g, draws['shot'], draws['read'])
    meta = dict(seed=seed, native_noisy_sha256=digest_tensor(noisy), sensor_calls=1,
                branch_sha256={name: digest_tensor(image) for name, image in branches.items()},
                saturation_fraction=float(aux['saturation_fraction'].mean()),
                dark_fraction=float(aux['dark_fraction'].mean()),
                temporal_irradiance_mae=float(optical['temporal_blur_mae'].mean()))
    return branches, meta, noisy[0, 0].cpu().numpy()


def project(points, position, rotation, width, height, fov):
    local = (points - position) @ rotation
    return torch.stack([(1 - local[..., 1] / local[..., 0] / fov) * width / 2 - .5,
                        (1 - local[..., 2] / local[..., 0] / (fov * height / width)) * height / 2 - .5], -1), local[..., 0] > 1e-5


@torch.no_grad()
def motion_scale(args, reference, state):
    i = state['episode']
    env = copy.deepcopy(reference)
    # Renderer expects batch dimensions to match; keep the existing 32-layout batch,
    # but repeat the selected geometry/state. No new trajectory is generated.
    for key in ('voxels', 'R_scene', 'R_scene_T', 'R_cam'):
        value = getattr(reference, key)
        setattr(env, key, value[i:i+1].expand_as(value).clone())
    for key in ('p', 'v', 'R', 'R_old'):
        setattr(env, key, state[key][None].expand_as(getattr(env, key)).clone())
    local_p = env.R_scene_T @ env.p[..., None]
    position = local_p[..., 0]
    velocity = (env.R_scene_T @ env.v[..., None])[..., 0]
    rotation = env.R_scene_T @ env.R
    omega = body_angular_velocity(env.R_old, env.R, 1 / args.base_control_freq)
    cam = rotation @ env.R_cam
    _, aux = render_scene_irradiance(env, position, cam)
    points = aux['points'][0].reshape(-1, 3)
    depth = aux['geometry_depth'][0].flatten()
    center, half = state['center'], float(reference.slit_half_y[i])
    edge_distance = torch.minimum((points[:, 1] - (center - half)).abs(),
                                  (points[:, 1] - (center + half)).abs())
    edge = (points[:, 0].abs() <= reference.simple_wall_half_x + .001) & (edge_distance <= .10) & ((points[:, 2] - 1.5).abs() <= .3)
    rows = []
    for exposure in EXPOSURES:
        duration = float(env.gray_camera.calibration.exposure_to_us(torch.tensor(exposure))) * 1e-6
        start_p = position[0] - velocity[0] * duration
        start_r = rotation[0] @ rotation_exp(-omega[:1] * duration)[0] @ env.R_cam[0]
        end_uv, end_valid = project(points, position[0], cam[0], 96, 72, args.fov_x_half_tan)
        start_uv, start_valid = project(points, start_p, start_r, 96, 72, args.fov_x_half_tan)
        travel = (end_uv - start_uv).norm(dim=-1)
        valid = (depth < 99) & start_valid & end_valid
        stats = {}
        for label, mask in (('visible_surfaces', valid), ('slit_edge', valid & edge)):
            values = travel[mask]
            stats[label] = {'points':int(mask.sum()), 'sensor_px_p50_p95_max':torch.quantile(values, values.new_tensor([.5,.95,1])).tolist() if len(values) else None}
            if len(values):
                stats[label]['input_48x36_px_p50_p95_max'] = [v * .5 for v in stats[label]['sensor_px_p50_p95_max']]
        rows.append(dict(exposure01=exposure, exposure_us=duration*1e6, displacement=stats))
    return dict(requested_x=state['requested_x'], actual_local_position=position[0].tolist(),
                velocity_m_s=state['v'].tolist(), speed_m_s=float(state['v'].norm()),
                body_angular_velocity_rad_s=omega[0].tolist(), angular_speed_rad_s=float(omega[0].norm()),
                body_rotation=state['R'].tolist(), step=state['step'], time_s=state['time_s'], rows=rows,
                interpretation='Endpoint image displacement of fixed visible surface points over the same constant-twist shutter model; not an exact blur kernel or measured optics.')


def evaluate_condition(args, reference, state, scenario, moving, exposure, gain):
    datasets, audit, representative = {}, {}, None
    for split in ('fit', 'check'):
        xs = {name: [] for name in ('48x36', '96x72')}
        labels = []
        for side in (-1, 1):
            branches, meta, image = samples(args, reference, state, scenario, moving, exposure, gain, split, side)
            for name, value in branches.items():
                xs[name].append(value.flatten(1))
            labels.append(torch.full((COUNT,), side, device=reference.device))
            audit[f'{split}_{side}'] = meta
            if split == 'check' and side == 1:
                representative = image
        datasets[split] = ({name:torch.cat(value) for name,value in xs.items()}, torch.cat(labels))
    rows = []
    for name in ('48x36', '96x72'):
        correct, scores = ridge_probe(datasets['fit'][0][name], datasets['fit'][1],
                                      datasets['check'][0][name], datasets['check'][1])
        rows.append(dict(requested_x=state['requested_x'], scenario=scenario,
            motion='moving' if moving else 'stationary', exposure01=exposure, gain01=gain,
            resolution=name, accuracy=float(correct.mean()), check_examples=2*COUNT,
            correct=correct.cpu().tolist(), scores=scores.cpu().tolist()))
    return rows, audit, representative


def screen_decision(rows):
    if rows and all(row['accuracy'] >= .95 for row in rows):
        return dict(category='probe_ceiling', label='这个检查太容易，无法判断', candidates=[], contrasts=[])
    contrasts, candidates = [], []
    for x, scenario, resolution in itertools.product(DISTANCES, SCENARIOS, ('48x36', '96x72')):
        lookup = {(r['motion'],r['exposure01']):r['accuracy'] for r in rows
                  if r['requested_x']==x and r['scenario']==scenario and r['resolution']==resolution}
        for short, long in itertools.combinations(EXPOSURES, 2):
            static = lookup['stationary',long] - lookup['stationary',short]
            moving = lookup['moving',long] - lookup['moving',short]
            interaction = static - moving
            # Screen changes in RELATIVE exposure benefit; a preference reversal
            # is not required. This is an engineering signal, not joint-control proof.
            candidate = abs(interaction) >= THRESHOLD and max(abs(static),abs(moving)) >= THRESHOLD
            contrast = dict(requested_x=x, scenario=scenario, resolution=resolution,
                short_exposure=short,long_exposure=long,stationary_long_minus_short=static,
                moving_long_minus_short=moving,interaction=interaction,candidate=candidate)
            contrasts.append(contrast)
            if candidate:candidates.append(contrast)
    native = any(c['resolution']=='96x72' for c in candidates)
    low = any(c['resolution']=='48x36' for c in candidates)
    if native and not low:
        category,label='native_only_tradeoff','只有大图出现候选取舍'
    elif low:
        category,label='current_input_tradeoff','当前48×36输入已出现候选取舍'
    else:
        category,label='no_tradeoff','两种大小均未发现预定尺度的曝光—运动取舍'
    return dict(category=category,label=label,candidates=candidates,contrasts=contrasts)


def matched_gain(calibration, short, long, gain=.02):
    if calibration.gain_mapping != 'log':
        raise ValueError('bounded gain follow-up requires the unchanged log mapping')
    ratio=float(calibration.exposure_to_us(torch.tensor(long))/calibration.exposure_to_us(torch.tensor(short)))
    factor=float(calibration.gain_to_factor(torch.tensor(gain)))*ratio
    if not calibration.gain_factor_min <= factor <= calibration.gain_factor_max:
        return dict(available=False,required_factor=factor,reason='outside existing gain range; no clamping or range extension')
    normalized=math.log(factor/calibration.gain_factor_min)/math.log(calibration.gain_factor_max/calibration.gain_factor_min)
    return dict(available=True,required_factor=factor,gain01=normalized,
                interpretation='Matched exposure-time times gain, not matched SNR or measured brightness')


def make_figure(out, rows, scales, images):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(14,12),layout='constrained')
    grid=fig.add_gridspec(4,6,height_ratios=[1.3,1.3,1,1])
    for index,(x,scenario) in enumerate(itertools.product(DISTANCES,SCENARIOS)):
        ax=fig.add_subplot(grid[index//2,(index%2)*3:(index%2+1)*3])
        for resolution,color in (('48x36','#256c92'),('96x72','#b37432')):
            for motion,style in (('stationary','--'),('moving','-')):
                values=[next(r['accuracy'] for r in rows if r['requested_x']==x and r['scenario']==scenario and r['resolution']==resolution and r['motion']==motion and r['exposure01']==e) for e in EXPOSURES]
                ax.plot(EXPOSURES,values,style+'o',color=color,label=f'{resolution} {motion}')
        scale=next(s for s in scales if s['requested_x']==x)
        ax.set(title=f'x≈{x} m, {scenario} | actual angular rate {scale["angular_speed_rad_s"]:.3f} rad/s',
               xlabel='Exposure command E',ylabel='Slit-position probe accuracy',ylim=(.4,1.03))
        ax.legend(frameon=False,fontsize=8,ncol=2)
    # Fixed illustration: close dark state, both motions and all three exposures.
    for ri,motion in enumerate(('stationary','moving')):
        for ci,e in enumerate(EXPOSURES):
            native=images[f'-0.4_dark_{motion}_{e}']
            low=F.adaptive_avg_pool2d(torch.from_numpy(native)[None,None],(36,48))[0,0].numpy()
            combined=np.concatenate([native,np.repeat(np.repeat(low,2,0),2,1)],axis=1)
            ax=fig.add_subplot(grid[ri+2,ci*2:ci*2+2])
            ax.imshow(combined,cmap='gray',vmin=0,vmax=1,interpolation='nearest');ax.axis('off')
            ax.set_title(f'{motion}, E={e:.2f} | 96x72 / 48x36',fontsize=9)
    fig.suptitle('Same noisy sensor image, two input sizes\nSingle-frame diagnostic only; provisional simulator; no policy training',fontsize=15)
    fig.savefig(out/'comparison.png',dpi=150)
    fig.savefig(out/'comparison_preview.jpg',dpi=65,pil_kwargs={'quality':78})
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out_dir',required=True)
    parser.add_argument('--device',default='cuda')
    cli=parser.parse_args()
    out=Path(cli.out_dir).resolve();out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    device=torch.device(cli.device)
    args=load_intent_args(CONFIG)
    protocol=dict(sensor_resolution_wh=[96,72],input_resolutions_wh=RESOLUTIONS,
        exposures=EXPOSURES,gain=.02,scenarios=SCENARIOS,requested_x=DISTANCES,
        fit_per_class=COUNT,check_per_class=COUNT,center_difference_m=CENTER_DIFFERENCE_M,
        interaction_threshold=THRESHOLD,ceiling_rule='all 48 accuracies >=95%',
        candidate_rule='absolute change in long-minus-short benefit >=10pp; absolute benefit in one arm >=10pp; reversal not required',
        conditional_followup='at most one highest-interaction candidate, deterministic order tie break; same nuisance data',
        no_policy_training=True,no_new_scene=True,no_sensor_changes=True,
        nuisance=dict(yaw_deg=[-.25,.25],texture_phase_rad=[0,2*math.pi],light_factor=[.92,1.08],
                      fit_seed=391000,check_seed=491000,class_offset='(class+1)*100'),
        limitations=['single successful privileged reference trajectory','static is a counterfactual image, not free stopping',
            'fixed-rule ridge fits are diagnostic only','not navigation success or learning-method comparison'])
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    reference,states,witness=replay_reference(args,device)
    (out/'reference_trajectory.json').write_text(json.dumps(witness,indent=2)+'\n')
    scales=[motion_scale(args,reference,state) for state in states]
    (out/'motion_scale.json').write_text(json.dumps(scales,indent=2)+'\n')
    rows,audits,images=[],[],{}
    for state,scenario,moving,exposure in itertools.product(states,SCENARIOS,(False,True),EXPOSURES):
        results,audit,image=evaluate_condition(args,reference,state,scenario,moving,exposure,.02)
        rows.extend(results)
        motion='moving' if moving else 'stationary'
        key=f'{state["requested_x"]}_{scenario}_{motion}_{exposure}'
        audits.append(dict(condition=key,images=audit));images[key]=image
        print(json.dumps({k:v for k,v in results[0].items() if k not in ('correct','scores')})+
              ' native_accuracy='+str(results[1]['accuracy']),flush=True)
    decision=screen_decision(rows)
    followup=dict(performed=False,reason='no candidate or probe ceiling; stop without expansion')
    if decision['candidates']:
        chosen=max(decision['candidates'],key=lambda c:abs(c['interaction']))
        followup=dict(candidate=chosen,**matched_gain(reference.gray_camera.calibration,
            chosen['short_exposure'],chosen['long_exposure']))
        followup['performed']=followup['available']
        if followup['available']:
            state=next(s for s in states if s['requested_x']==chosen['requested_x'])
            extra=[]
            for moving in (False,True):
                results,audit,_=evaluate_condition(args,reference,state,chosen['scenario'],moving,
                    chosen['short_exposure'],followup['gain01'])
                extra.extend(results);audits.append(dict(condition='conditional_gain_'+str(moving),images=audit))
            followup['rows']=extra
    result=dict(protocol=protocol,calibrated=False,decision=decision,rows=rows,
                gain_followup=followup,motion_scale=scales,selected_reference_episode=witness['selected_episode'],
                claim='Input-scale and exposure/motion cue diagnostic; no learned joint-control or hardware claim.')
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'image_audit.json').write_text(json.dumps(audits,indent=2)+'\n')
    np.savez_compressed(out/'representative_frames.npz',**images)
    make_figure(out,rows,scales,images)
    write_run_provenance(out,args)
    (out/'sha256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in out.iterdir() if p.is_file()},indent=2)+'\n')
    print('DECISION',decision['category'],decision['label'],flush=True)

if __name__=='__main__':
    main()
