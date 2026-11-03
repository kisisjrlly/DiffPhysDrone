#!/usr/bin/env python3
"""Gated real-network sensing/decision diagnostic. No joint-policy training.

Stage one is mandatory. A failed flight baseline stops intervention/gradient
claims; the only permitted training is one fixed-camera 600-update adaptation.
All scores use the locked historical layouts, not independent generalization.
"""
import argparse
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from intent_model import build_policy
from rollout_ops import (build_local_frame, build_state_vector, compute_target_velocity,
    decode_action_direct, init_camera_params, policy_forward, update_camera_params)
from tools.audit_slit_information_deadline import (
    HistoricalBatch, SPLITS, SCENARIOS, STEPS, FPS, observe, sha, write_json)
from tools.intent_common import load_intent_args
from train_utils import write_run_provenance

CHECKPOINT = Path('/home/zhaoguodong/work/code/DiffPhysDrone/checkpoint/2026-09-24-10-33-44/final.pth')
CHECKPOINT_SHA = '748d5e11fcf4ed66af462e80c2787d7aa798ae25e37227aa31b955fcdb621b7a'
CONFIG = 'configs/gray_joint_slit.args'


def flight_args():
    return load_intent_args(CONFIG, policy_arch='legacy', joint_backward='legacy',
        grad_decay=.3, observation_grad_mode='detached', camera_control_mode='fixed',
        fixed_camera_exposure=.10, fixed_camera_gain=0., sensor_grad_mode='detached')


def parameter_hash(model, camera):
    h = hashlib.sha256()
    for name, value in model.named_parameters():
        if model._is_camera_parameter(name) == camera:
            h.update(name.encode())
            h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_model(checkpoint, device='cuda'):
    model = build_policy(flight_args()).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device), strict=True)
    model.eval()
    return model


def baseline_gate(normal, zero):
    """Only development scores may decide whether intervention is interpretable."""
    if normal == 8:
        return 'ceiling_stop'
    return 'pass' if normal >= 4 and normal - zero >= 2 else 'needs_flight_baseline'


class NetworkRun:
    """Restorable complete batch state, including memories, images and queues.

    Snapshot before observation at a control step. Deepcopy preserves camera
    queue metadata too; Env.save_state alone is insufficient for this purpose.
    Model has no episode state beyond the explicit recurrent tensors.
    """
    def __init__(self, history, scenario):
        self.env = history.new_env(scenario)
        self.env.camera_control_mode = 'learned'
        self.env.fixed_camera_exposure, self.env.fixed_camera_gain = .10, 0.
        self.e, self.g = init_camera_params(self.env, 32, self.env.device)
        self.queue = [self.env.act.clone(), self.env.act.clone()]
        self.h = self.cam_h = self.previous = None
        self.step = 0
        self.alive = torch.ones(32, dtype=torch.bool, device=self.env.device)
        self.success = torch.zeros_like(self.alive)
        self.collision = torch.zeros_like(self.alive)
        self.cross_step = torch.full((32,), -1, device=self.env.device, dtype=torch.long)
        self.terminal_step = torch.full_like(self.cross_step, 89)
        self.time = self.e.new_full((32,), 6.)
        self.minimum = self.e.new_full((32,), 100.)
        self.effort = self.e.new_zeros(32)

    def snapshot(self):
        return copy.deepcopy(self)

    @torch.no_grad()
    def advance(self, history, model, zero=False, camera=(.10, 0.), offset=(0., 0., 0.)):
        env, step, args = self.env, self.step, flight_args()
        _, gray = observe(history, env, range(32), self.e, self.g, step)
        previous = gray if self.previous is None else self.previous
        pixels = torch.cat([gray, previous], 1)
        if zero:
            pixels = torch.zeros_like(pixels)
        self.previous = gray
        frame = build_local_frame(env)
        velocity_target = compute_target_velocity(env.p_target-env.p.detach(), env)
        state, _, cs, cms = build_state_vector(env, velocity_target, frame, self.e, self.g,
            args.no_odom, args.include_camera_state_in_obs)
        raw, _, self.h, self.cam_h = policy_forward(model, state, self.h, env,
            gray_obs=pixels, cam_hx=self.cam_h, camera_state=cs, camera_motion_state=cms,
            pending_flight_commands=self.queue)
        command = decode_action_direct(raw + raw.new_tensor(offset), frame, env, 32, args.max_acc_cmd)
        executed = self.queue.pop(0)
        self.queue.append(command)
        target = self.e.new_tensor(camera)[None].expand(32, -1)
        duration = env.gray_camera.calibration.exposure_to_us(self.e) * 1e-6
        row = dict(step=step, position=env.p.cpu().tolist(), velocity=env.v.cpu().tolist(),
            rotation=env.R.cpu().tolist(), applied_eg=torch.stack([self.e,self.g],-1).cpu().tolist(),
            requested_eg=target.cpu().tolist(), submitted_command=command.cpu().tolist(),
            executed_command=executed.cpu().tolist(),
            exposure_start_s=(step/FPS-duration).cpu().tolist(), exposure_end_s=step/FPS,
            command_effective_frame=step+env._camera_effective_delay_frames+1,
            alive=self.alive.cpu().tolist())
        self.e, self.g, _ = update_camera_params(target,self.e,self.g,env)
        self.effort += self.alive*command.square().sum(-1)/FPS
        history.run_step(env, executed, step)
        clearance = env.transition_clearance()
        hit = env.collision_since_last_step() | (clearance <= args.collision_clearance)
        reached = (env.p-env.p_target).norm(dim=-1) < .35
        local_x = (env.R_scene_T @ env.p[...,None])[:,0,0]
        crossed = local_x >= args.simple_wall_x+env.simple_wall_half_x+env.drone_radius
        first = self.alive & ~hit & crossed & (self.cross_step<0)
        self.cross_step[first] = step
        self.minimum = torch.minimum(self.minimum,torch.where(self.alive,clearance,self.minimum))
        self.collision |= self.alive & hit
        self.success |= self.alive & ~hit & reached
        done = self.alive & (hit | reached)
        self.terminal_step[done] = step
        self.time[done] = (step+1)/FPS
        self.alive &= ~done
        self.step += 1
        row.update(clearance=clearance.cpu().tolist(), success=self.success.cpu().tolist(),
                   collision=self.collision.cpu().tolist())
        return row

    def episodes(self, indices, scenario):
        return [dict(episode_index=i, scenario=scenario, success=bool(self.success[i]),
            collision=bool(self.collision[i]),timeout=bool(self.alive[i]),
            terminal_step=int(self.terminal_step[i]), cross_step=int(self.cross_step[i]),
            terminal_time_s=float(self.time[i]),
            scored_time_s=float(self.time[i]) if self.success[i] else 6.,
            control_cost=float(self.effort[i]),min_clearance_m=float(self.minimum[i])) for i in indices]


def run_baseline(history, model, zero=False):
    episodes, timelines = [], {}
    for scenario in SCENARIOS:
        run = NetworkRun(history,scenario)
        trace=[]
        for _ in range(STEPS):
            trace.append(run.advance(history,model,zero=zero))
            if not run.alive[list(SPLITS['development'])].any():
                break
        episodes.extend(run.episodes(SPLITS['development'],scenario))
        timelines[scenario]=trace
    return dict(episodes=episodes,timelines=timelines,
        successes=sum(r['success'] for r in episodes), zero_pixels=zero)


def evaluate(checkpoint, out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    if (out/'result.json').exists():
        existing=json.loads((out/'result.json').read_text())
        if existing['checkpoint_sha256'] != sha(checkpoint) or existing.get('compatible_tool_sha256',existing['source_sha256']) != sha(__file__):
            raise RuntimeError('Refusing stale results from a different checkpoint/tool source')
        return existing
    history=HistoricalBatch('cuda')
    model=load_model(checkpoint)
    normal=run_baseline(history,model)
    zero=run_baseline(history,model,True)
    result=dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),source_sha256=sha(__file__),
        normal_success=normal['successes'],zero_success=zero['successes'],
        gate=baseline_gate(normal['successes'],zero['successes']),
        parameter_count=sum(p.numel() for p in model.parameters()),
        camera_parameter_hash=parameter_hash(model,True),flight_parameter_hash=parameter_hash(model,False),
        layouts=SPLITS, scenarios=SCENARIOS,
        limitation='Historical tracking-success-selected development layouts; not independent test data.')
    write_json(out/'normal.json',normal);write_json(out/'zero.json',zero)
    write_json(out/'result.json',result)
    print(json.dumps(result,indent=2),flush=True)
    return result


def adaptation_tokens(out):
    import shlex
    tokens=shlex.split(Path(CONFIG).read_text(),comments=True)
    tokens += ['--policy_arch','legacy','--joint_backward','legacy','--grad_decay','.3',
        '--observation_grad_mode','detached','--camera_control_mode','fixed',
        '--fixed_camera_exposure','.10','--fixed_camera_gain','0',
        '--sensor_grad_mode','detached','--train_flight_only','--resume',str(CHECKPOINT),
        '--checkpoint_dir',str(out),'--num_iters','600','--batch_size','8','--lr','1e-5',
        '--seed','42','--deterministic','--no-amp','--wandb_disabled',
        '--coef_v','10.5','--coef_collide','6','--coef_d_acc','.05','--coef_d_jerk','.05',
        '--coef_cam_smooth','0','--coef_obj_avoidance','0','--coef_terminal_goal','0','--scenarios','dark','bright','dark_to_bright','bright_to_dark']
    return tokens


def adapt(out):
    """One final checkpoint, never best-checkpoint selection or a retry."""
    out=Path(out)
    if out.exists():
        raise RuntimeError('Adaptation already started: refusing another run; inspect its saved artifacts')
    out.mkdir(parents=True)
    write_json(out/'single_run_authorization.json',dict(updates=600,seed=42,batch=8,lr=1e-5,
        checkpoint_sha256=sha(CHECKPOINT),command=[sys.executable,__file__,'_train',str(out)]))
    with (out/'console.log').open('w') as log:
        result=subprocess.run([sys.executable,__file__,'_train',str(out)],stdout=log,stderr=subprocess.STDOUT,cwd=ROOT)
    if result.returncode:
        write_json(out/'aborted.json',dict(returncode=result.returncode,no_retry=True))
        raise RuntimeError(f'Adaptation aborted, no retry: {out}/console.log')
    before=load_model(CHECKPOINT); after=load_model(out/'final.pth')
    if parameter_hash(before,True)!=parameter_hash(after,True):
        raise RuntimeError('Frozen camera weights changed')
    metrics=[json.loads(line) for line in (out/'metrics.jsonl').read_text().splitlines()]
    if len(metrics)!=600 or metrics[-1]['iteration']!=600:
        raise RuntimeError('Adaptation did not complete exactly 600 updates')
    write_json(out/'verified.json',dict(updates=600,camera_unchanged=True,
        camera_hash=parameter_hash(after,True),final_sha256=sha(out/'final.pth')))


def train_worker(out):
    # Fail fast on a nonfinite loss or gradient. The legacy trainer otherwise
    # skips nonfinite losses; this bounded run must stop rather than retry.
    import trainer
    original_loss=trainer._loss_from_rollout
    def checked_loss(*args,**kwargs):
        result=original_loss(*args,**kwargs)
        if not torch.isfinite(result[0]):
            raise FloatingPointError('Nonfinite loss: bounded adaptation stopped')
        return result
    trainer._loss_from_rollout=checked_loss
    original_step=torch.optim.AdamW.step
    def checked_step(optimizer,*args,**kwargs):
        if any(p.grad is not None and not torch.isfinite(p.grad).all()
               for group in optimizer.param_groups for p in group['params']):
            raise FloatingPointError('Nonfinite gradient: bounded adaptation stopped')
        return original_step(optimizer,*args,**kwargs)
    torch.optim.AdamW.step=checked_step
    import main_cuda
    sys.argv=['main_cuda.py',*adaptation_tokens(out)]
    from config import build_parser, parse_scenarios, validate_args
    resolved=build_parser().parse_args(sys.argv[1:])
    resolved.scenarios=parse_scenarios(resolved.scenarios)
    validate_args(resolved)
    original=json.loads(CHECKPOINT.with_name('args.json').read_text())
    for key in ('coef_v','coef_collide','coef_obj_avoidance','coef_d_acc',
                'coef_d_jerk','coef_cam_smooth','grad_decay','loss_v_window'):
        if getattr(resolved,key)!=original[key]:
            raise RuntimeError(f'Historical loss mismatch: {key}')
    if resolved.coef_terminal_goal!=0:
        raise RuntimeError('No additional terminal objective is permitted')
    main_cuda.main()


def main():
    if len(sys.argv)>1 and sys.argv[1]=='_train':
        train_worker(Path(sys.argv[2]));return
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out_dir',type=Path,required=True)
    p.add_argument('--stage',choices=['baseline','adapt','evaluate_adapted'],required=True)
    args=p.parse_args()
    torch.set_num_threads(4)
    if sha(CHECKPOINT)!=CHECKPOINT_SHA:
        raise RuntimeError('Frozen primary checkpoint hash mismatch')
    args.out_dir.mkdir(parents=True,exist_ok=True)
    if args.stage=='baseline':
        write_run_provenance(str(args.out_dir),flight_args())
        evaluate(CHECKPOINT,args.out_dir/'baseline')
    elif args.stage=='adapt':
        gate=json.loads((args.out_dir/'baseline/result.json').read_text())['gate']
        if gate!='needs_flight_baseline':raise RuntimeError('Adaptation gate is not open')
        adapt(args.out_dir/'adaptation')
    else:
        if not (args.out_dir/'adaptation/verified.json').exists():
            raise RuntimeError('Only the verified final 600-update checkpoint can be evaluated')
        evaluate(args.out_dir/'adaptation/final.pth',args.out_dir/'adapted_baseline')

if __name__=='__main__':main()
