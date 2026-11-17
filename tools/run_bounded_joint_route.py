#!/usr/bin/env python
"""Resumeable bounded image-driven experiment. See locked protocol in output."""
from __future__ import annotations
import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from bounded_route.core import *
from bounded_route.integrity import verify_prepared, verify_reader_metadata, verify_checkpoint

ARMS=('A1','A2','A3','B','C','D')
MODES=dict(A1='mean',A2='gradient',A3='motion',B='learned',C='learned',D='learned')
SPLITS=dict(collection=list(range(128)), train=list(range(256)), development=list(range(16)), confirmation=list(range(64)))


def sync_directory(path):
    descriptor=os.open(str(Path(path)),os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(descriptor)
    finally:os.close(descriptor)


def durable_append(path,value):
    with Path(path).open('a') as stream:
        stream.write(json.dumps(value)+'\n');stream.flush();os.fsync(stream.fileno())


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    with temp.open('w') as stream:
        stream.write(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        stream.flush();os.fsync(stream.fileno())
    temp.replace(path)
    sync_directory(path.parent)


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    with tmp.open('wb') as stream:
        torch.save(value,stream);stream.flush();os.fsync(stream.fileno())
    tmp.replace(path);sync_directory(path.parent)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_files():
    tracked=subprocess.check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()
    extras=['bounded_route/core.py','bounded_route/__init__.py','tools/run_bounded_joint_route.py',
            'configs/gray_bounded_joint.args','configs/gray_single_wall_slit_diff.args','bounded_route/report.py','bounded_route/integrity.py',
            'configs/calibration/grayscale_research_50ms.json','tests/test_bounded_joint_route.py']
    return sorted(set(p for p in tracked+extras if (ROOT/p).is_file() and
                      (p.endswith('.py') or p.endswith('.args') or p.endswith('.json') or p=='AGENTS.md')))


def source_hashes():
    return {p:sha(ROOT/p) for p in source_files()}


class Ledger:
    def __init__(self,out):
        self.path=out/'ledger.jsonl'; self.current=None; self.started=None

    def rows(self):
        return [json.loads(s) for s in self.path.read_text().splitlines()] if self.path.exists() else []

    def seconds(self):
        return sum(r.get('seconds',0) for r in self.rows()) + (time.monotonic()-self.started if self.started is not None else 0)

    @contextlib.contextmanager
    def charge(self,kind,arm=None):
        if self.seconds()>=24*3600:
            raise RuntimeError('24 GPU-hour total allocation exhausted')
        if self.current is not None:
            yield;return
        start=time.monotonic();self.started=start;self.current=(kind,arm)
        try:
            yield
        finally:
            if torch.cuda.is_available():torch.cuda.synchronize()
            row=dict(kind=kind,arm=arm,seconds=time.monotonic()-start,time=time.time())
            durable_append(self.path,row)
            self.current=None;self.started=None

    def arm_seconds(self,arm):
        return sum(r.get('seconds',0) for r in self.rows() if r.get('arm')==arm) + (time.monotonic()-self.started if self.current and self.current[1]==arm else 0)

    def episode(self,namespace,ids,kind,arm):
        # A planned full rollout is counted even if an exception aborts the call.
        row=dict(kind='rollout',phase=kind,arm=arm,namespace=namespace,ids=list(ids),count=len(ids),time=time.time())
        durable_append(self.path,row)


class Experiment:
    def __init__(self,args):
        self.args=args;self.out=Path(args.out_dir).resolve();self.out.mkdir(parents=True,exist_ok=True)
        torch.set_num_threads(4)
        self.device=torch.device(args.device)
        self.ledger=Ledger(self.out)
        self.factory=EpisodeFactory(str(ROOT/args.config),self.device)
        self.reader=Reader().to(self.device)
        self.runner=Runner(self.factory,self.reader)
        self.generator=torch.Generator(device='cpu').manual_seed(42)
        self.lock=self.out/'protocol.json'
        if self.lock.exists():
            old=json.loads(self.lock.read_text())
            hashes=source_hashes()
            differences=[p for p in set(old['source_hashes'])|set(hashes) if old['source_hashes'].get(p)!=hashes.get(p)]
            if differences:raise RuntimeError(f'Source hash mismatch; do not mix evidence: {differences}')
            if old['config_hash']!=sha(ROOT/args.config):raise RuntimeError('Configuration hash mismatch')
            if old['splits']!=SPLITS:raise RuntimeError('Episode split mismatch')
        else:
            self.snapshot()
        verify_prepared(self.out)
        if (self.out/'reader.pth').exists():
            data=torch.load(self.out/'reader.pth',map_location=self.device,weights_only=False)
            verify_reader_metadata(self.out,data)
            self.reader.load_state_dict(data['model'],strict=True)
            self.reader.eval().requires_grad_(False)

    def snapshot(self):
        hashes=source_hashes()
        for p in hashes:
            target=self.out/'source'/p;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(ROOT/p,target)
        write_json(self.lock,dict(schema=1,source_hashes=hashes,config_hash=sha(ROOT/self.args.config),
            actual_args=vars(self.factory.args),splits=SPLITS,seed=42,limits=dict(gpu_hours=24,prepare_hours=4,
            training_hours=12,evaluation_hours=3.2,repair_hours=4.8,rollouts_per_arm=4096,maximum_rounds=7),
            training_semantics='one frozen shared reader; no truth in policy; same full task cost',
            motion_bounds=[MOTION_LOW,MOTION_HIGH],motion_initial=MOTION_INITIAL,
            camera_normalized_bounds=[-1,1],camera_coefficient_scale=2,
            pose_backward='warp_ste approximate',saturation='hard forward and derivative',
            physical=dict(exposure_us=[100,50000],gain=[1,8],initial_exposure_us=890,initial_gain=1,
                          body_radius=.12,slit_half_width=.15,clearance=.0011),
            environment=dict(python=sys.executable,torch=torch.__version__,cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else None,
                git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True))))
        write_json(self.out/'gradient_paths.json',dict(
            exposure_to_sensor='physical shutter units, once-only reparameterized noise, actual hard clamp derivative',
            exposure_to_sampling='history sampling retains duration derivative; nine samples',
            reader='weights frozen; input and attitude derivatives retained',
            camera_feedback='same mean-AE recurrence retains graph; no hidden detach',
            flight_queue='two commands; actual executed input',
            dynamics='grad_decay=1; differentiable attitude/drag, explicit noise',
            pose_to_image='warp_ste; local fixed-depth approximation, misses disocclusion',
            terminal='hard event active masks; no exact event-time derivative claim'))

    def run(self,namespace,ids,m,c,mode,*,phase,arm=None,grad=False,**kwargs):
        self.ledger.episode(namespace,ids,phase,arm)
        with torch.set_grad_enabled(grad):
            return self.runner.rollout(namespace,ids,m,c,mode,**kwargs)

    def evaluate(self,arm,m,c,namespace='development',record=False,mode=None,tag=None):
        rows,traces=[],[]
        for j in range(0,len(SPLITS[namespace]),4):
            ids=SPLITS[namespace][j:j+4]
            _,ep,tr,_=self.run(namespace,ids,m,c,mode or MODES[arm],phase='evaluation',arm=arm,record=record)
            rows.extend(ep)
            if record:traces.append(dict(ids=ids,timeline=tr))
        result=dict(arm=arm,namespace=namespace,episodes=rows,successes=sum(x['success'] for x in rows),
                    collisions=sum(x['collision'] for x in rows),time=sum(x['time_s'] for x in rows)/len(rows),
                    control=sum(x['control'] for x in rows)/len(rows),
                    motion=m.detach().cpu().tolist(),camera=c.detach().cpu().tolist(),
                    reader_hash=sha(self.out/'reader.pth'),elapsed_seconds=self.ledger.seconds(),arm_seconds=self.ledger.arm_seconds(arm))
        if record:save(self.out/'timelines'/f'{tag or arm}_{namespace}.pth',traces)
        return result

    def prepare(self):
        if (self.out/'prepared.json').exists():return
        with self.ledger.charge('prepare'):
            dataset=self.out/'reader_data.pth'
            if not dataset.exists():
                images,rots,labels,episodes,xs,exps=[],[],[],[],[],[]
                settings=[(100,1),(890,1),(8000,1),(50000,1),(100,8),(890,8),(8000,8),(50000,8)]
                for i in range(128):
                    cache=self.out/'collection'/f'{i:03}.pth'
                    if cache.exists():
                        item=torch.load(cache,weights_only=False)
                    else:
                        speed=(.3,.6,.9,1.15)[(i//4)%4]
                        setting=settings[i//16]
                        _,ep,_,data=self.run('collection',[i],initial_motion(self.device),torch.zeros(6,device=self.device),
                            'mean',phase='collection',teacher=True,collect=True,teacher_speed=speed,
                            fixed_camera=setting,teacher_bias=.15*(1 if (i//4)%2 else -1))
                        item=dict(data=data,episode=ep[0]);save(cache,item)
                    for im,ro,la,al,x,t in item['data']:
                        if bool(al[0]):
                            images.append(im);rots.append(ro);labels.append(la);episodes.append(i);xs.append(x);exps.append(t)
                    print('COLLECT',i,item['episode']['success'],flush=True)
                data=dict(images=torch.cat(images),rotations=torch.cat(rots),labels=torch.cat(labels),
                          episodes=torch.tensor(episodes),x=torch.cat(xs),exposure_us=torch.cat(exps))
                save(dataset,data)
            data=torch.load(dataset,weights_only=False)
            val_ids=[i for i in range(128) if i%4==(((i//4)%4)+(i//16))%4]
            isval=torch.isin(data['episodes'],torch.tensor(val_ids))
            write_json(self.out/'reader_split.json',dict(training=[i for i in range(128) if i not in val_ids],
                validation=val_ids,selection='trajectory split; scenario=(speed_index+camera_index)%4',
                data_hash=sha(dataset)))
            torch.manual_seed(42)
            self.reader=Reader().to(self.device);self.runner.reader=self.reader
            opt=torch.optim.Adam(self.reader.parameters(),lr=1e-3)
            gen=torch.Generator().manual_seed(42)
            train_ids=(~isval).nonzero().flatten()
            val=isval.nonzero().flatten()
            best=float('inf');curve=[];start_update=0
            reader_progress=self.out/'reader_progress.pth'
            if reader_progress.exists():
                progress=torch.load(reader_progress,map_location=self.device,weights_only=False)
                self.reader.load_state_dict(progress['model']);opt.load_state_dict(progress['optimizer'])
                gen.set_state(progress['rng'].cpu());best=progress['best'];curve=progress['curve'];start_update=progress['update']
            for k in range(start_update,1000):
                ids=train_ids[torch.randint(len(train_ids),(64,),generator=gen)]
                pred=self.reader(data['images'][ids].to(self.device),data['rotations'][ids].to(self.device))
                loss=F.huber_loss(pred,data['labels'][ids].to(self.device),delta=.03)
                opt.zero_grad();loss.backward();opt.step()
                if (k+1)%100==0:
                    self.reader.eval()
                    with torch.no_grad():
                        errors=[]
                        for st in range(0,len(val),128):
                            vi=val[st:st+128]
                            p=self.reader(data['images'][vi].to(self.device),data['rotations'][vi].to(self.device))
                            errors.append((p.cpu()-data['labels'][vi]).square())
                        mse=torch.cat(errors).mean().item()
                    curve.append(dict(update=k+1,validation_rmse=math.sqrt(mse)))
                    if mse<best:
                        best=mse;save(self.out/'reader.pth',dict(model=self.reader.state_dict(),update=k+1,
                            validation_mse=mse,data_hash=sha(dataset),split_hash=sha(self.out/'reader_split.json')))
                    print('READER',curve[-1],flush=True);self.reader.train()
                save(reader_progress,dict(model=self.reader.state_dict(),optimizer=opt.state_dict(),rng=gen.get_state(),best=best,curve=curve,update=k+1))
            state=torch.load(self.out/'reader.pth',weights_only=False,map_location=self.device)
            self.reader.load_state_dict(state['model'],strict=True);self.reader.eval().requires_grad_(False)
            write_json(self.out/'reader_curve.json',curve)
            self.reader_diagnostics(data,isval)
            self.integration_checks()
            write_json(self.out/'prepared.json',dict(reader_hash=sha(self.out/'reader.pth'),
                data_hash=sha(dataset),seconds=self.ledger.seconds(),complete=True))

    def reader_diagnostics(self,data,isval):
        with torch.no_grad():
            preds=[]
            for s in range(0,len(data['images']),128):
                preds.append(self.reader(data['images'][s:s+128].to(self.device),data['rotations'][s:s+128].to(self.device)).cpu())
        err=torch.cat(preds)-data['labels']
        output=[]
        for split,mask in [('fit',~isval),('validation',isval)]:
            for dlo,dhi in [(0,.5),(.5,1),(1,1.6),(1.6,4)]:
                for elo,ehi in [(0,500),(500,2000),(2000,10000),(10000,50001)]:
                    ids=mask & (-data['x']>=dlo)&(-data['x']<dhi)&(data['exposure_us']>=elo)&(data['exposure_us']<ehi)
                    if ids.any():output.append(dict(split=split,distance=[dlo,dhi],exposure_us=[elo,ehi],
                        count=int(ids.sum()),rmse=float(err[ids].square().mean().sqrt()),mae=float(err[ids].abs().mean())))
        write_json(self.out/'reader_error_groups.json',output)

    def integration_checks(self):
        m=initial_motion(self.device);c=torch.zeros(6,device=self.device)
        rows={}
        for tag,kwargs in [('normal',{}),('zero',dict(zero=True)),('teacher',dict(teacher=True))]:
            result=self.run('development',[0,1,2,3],m,c,'mean',phase='integration',record=True,**kwargs)
            rows[tag]=result[1]
            save(self.out/'integration'/f'{tag}.pth',result[2])
        a=self.run('integration',[0,1],m,c,'learned',phase='invariance',record=False)
        b=self.run('integration',[0,1],m,c,'learned',phase='invariance',record=True)
        assert a[1]==b[1],'Recording changes episode results'
        single=self.run('integration',[1],m,c,'learned',phase='invariance')[1][0]
        for key in ('success','collision','timeout','crossed'):
            assert single[key]==a[1][1][key],f'Batch composition changes {key}'
        for key in ('loss','control','final_distance','min_clearance'):
            assert math.isclose(single[key],a[1][1][key],rel_tol=1e-5,abs_tol=1e-6),f'Batch numeric divergence: {key}'
        torch.cuda.synchronize() if self.device.type=='cuda' else None
        t=time.monotonic()
        param=torch.zeros(6,device=self.device,requires_grad=True)
        loss=self.run('development',[0,1,2,3],m,param,'learned',phase='gradient_timing',grad=True)[0].mean()
        loss.backward()
        torch.cuda.synchronize() if self.device.type=='cuda' else None
        seconds_grad=time.monotonic()-t
        gradient=param.grad.detach().cpu()
        assert torch.isfinite(gradient).all(),'Nonfinite camera parameter gradient'
        t=time.monotonic()
        self.run('development',[0,1,2,3],m,c,'mean',phase='trial_timing')
        torch.cuda.synchronize() if self.device.type=='cuda' else None
        seconds_trial=time.monotonic()-t
        finite=[]
        for j in range(6):
            d=c.clone();d[j]=.01
            plus=self.run('development',[0,1,2,3],m,c+d,'learned',phase='finite_difference')[0].mean()
            minus=self.run('development',[0,1,2,3],m,c-d,'learned',phase='finite_difference')[0].mean()
            finite.append(float((plus-minus)/.02))
        estimate_round=3*120*seconds_trial + 120*seconds_grad + (96*seconds_trial+24*seconds_grad) + 120*seconds_trial
        # 20% runtime cushion, includes 24 development batches per common round.
        estimate_round=1.2*(estimate_round+24*seconds_trial)
        rounds=max(1,min(7,math.floor(12*3600/estimate_round)))
        write_json(self.out/'integration.json',dict(episodes=rows,record_and_batch_invariant=True,
            camera_gradient=gradient.tolist(),finite_difference=finite,
            gradient_seconds_per_batch=seconds_grad,trial_seconds_per_batch=seconds_trial,
            estimated_seconds_per_common_round=estimate_round,locked_rounds=rounds,
            interpretation='Finite differences include visibility/events; warp derivative remains approximate. No performance gate.',
            round_cap_reason='At most seven 480-rollout rounds, reserving all crossed evaluations/diagnostics inside 4096 charged episodes per arm.'))
        print('INTEGRATION',seconds_grad,seconds_trial,'ROUNDS',rounds,flush=True)

    def initial_state(self,arm):
        m=torch.nn.Parameter(initial_motion(self.device));c=torch.nn.Parameter(torch.zeros(6,device=self.device))
        indices = list(range(6)) if arm not in ('B','D') else (list(range(12)) if arm=='D' else [])
        esparam=torch.nn.Parameter(torch.cat([m.detach(),c.detach()]) if arm=='D' else m.detach().clone())
        return m,c,esparam,torch.optim.Adam([esparam],lr=.02),torch.optim.Adam([c],lr=.01)

    def optimize(self,arm,target_round):
        path=self.out/'training'/f'{arm}.pth'
        m,c,p,es,adam=self.initial_state(arm)
        cycle=0;rollouts=0;cursor=0;best=None;curve=[];round_done=0
        gen=torch.Generator().manual_seed(42)
        if path.exists():
            state=torch.load(path,map_location=self.device,weights_only=False)
            verify_checkpoint(self.out,state,sha(ROOT/self.args.config))
            m.data.copy_(state['motion']);c.data.copy_(state['camera']);p.data.copy_(state['esparam'])
            es.load_state_dict(state['es_optimizer']);adam.load_state_dict(state['camera_optimizer'])
            cycle,rollouts,cursor=state['cycle'],state['rollouts'],state['cursor']
            best,curve,round_done=state['best'],state['curve'],state['round_done']
            gen.set_state(state['rng'].cpu())
        per_cycle=4 if arm=='B' else (20 if arm=='C' else 16)
        cycles_round=480//per_cycle
        def next_ids():
            nonlocal cursor
            # Balanced groups; one common deterministic order independent of arm.
            ids=[(cursor+j)%256 for j in range(4)];cursor+=4;return ids
        def checkpoint():
            save(path,dict(motion=m.detach(),camera=c.detach(),esparam=p.detach(),es_optimizer=es.state_dict(),
                camera_optimizer=adam.state_dict(),cycle=cycle,rollouts=rollouts,cursor=cursor,
                best=best,curve=curve,round_done=round_done,rng=gen.get_state(),
                reader_hash=sha(self.out/'reader.pth'),config_hash=sha(ROOT/self.args.config)))
        with self.ledger.charge('train',arm):
            while round_done<target_round:
                while cycle < (round_done+1)*cycles_round:
                    if arm!='B':
                        sigma=.1+(.02-.1)*min(1.,rollouts/max(480,480*self.rounds))
                        totalgrad=torch.zeros_like(p)
                        for direction in range(2):
                            noise=torch.randn(p.shape,generator=gen).to(self.device)
                            ids=next_ids()
                            values=[]
                            # Project symmetric trials to legal bounds. Effective displacement is logged by parameters.
                            for sign in (1,-1):
                                trial=p.detach()+sign*sigma*noise
                                tm=trial[:6].clamp(0,1)
                                tc=trial[6:].clamp(-1,1) if arm=='D' else c.detach()
                                loss=self.run('train',ids,tm,tc,MODES[arm],phase='optimization',arm=arm)[0].mean()
                                values.append(loss)
                            totalgrad += (values[0]-values[1])/(2*sigma)*noise/2
                        es.zero_grad();p.grad=totalgrad;es.step()
                        with torch.no_grad():
                            p[:6].clamp_(0,1);m.copy_(p[:6])
                            if arm=='D':p[6:].clamp_(-1,1);c.copy_(p[6:])
                    if arm in ('B','C'):
                        ids=next_ids();adam.zero_grad()
                        # Full horizon, effective batch four; split only if memory requires it.
                        batch=self.args.gradient_microbatch
                        for start in range(0,4,batch):
                            sub=ids[start:start+batch]
                            loss=self.run('train',sub,m.detach(),c,'learned',phase='optimization',arm=arm,grad=True)[0].sum()/4
                            loss.backward()
                        if not torch.isfinite(c.grad).all():raise RuntimeError('Nonfinite learning gradient; no silent skip')
                        torch.nn.utils.clip_grad_norm_([c],1.)
                        adam.step()
                        with torch.no_grad():c.clamp_(-1,1)
                    rollouts+=per_cycle;cycle+=1
                    checkpoint()
                    if cycle%5==0:print('TRAIN',arm,'round',round_done+1,'rollouts',rollouts,flush=True)
                result=self.evaluate(arm,m.detach(),c.detach())
                result['optimization_rollouts']=rollouts;result['round']=round_done+1
                curve.append(result)
                rank=lambda r:(-r['successes'],r['time'],r['control'],r['round'])
                if best is None or rank(result)<rank(best):best=result
                round_done+=1;checkpoint()
                write_json(self.out/'training'/f'{arm}_curve.json',curve)
                print('ROUND',arm,round_done,result['successes'],'/16',flush=True)

    def train(self):
        if not (self.out/'prepared.json').exists():raise RuntimeError('Run prepare first')
        self.rounds=json.loads((self.out/'integration.json').read_text())['locked_rounds']
        if self.args.rounds is not None and self.args.rounds!=self.rounds:
            raise ValueError('Rounds locked by timing; cannot change after observing results')
        for rnd in range(1,self.rounds+1):
            for arm in ARMS:self.optimize(arm,rnd)
        write_json(self.out/'trained.json',dict(rounds=self.rounds,arms=ARMS,complete=True))

    def evaluate_all(self):
        if not (self.out/'trained.json').exists():raise RuntimeError('All locked training arms required')
        with self.ledger.charge('evaluate'):
            locked={}
            for arm in ARMS:
                state=torch.load(self.out/'training'/f'{arm}.pth',map_location='cpu',weights_only=False)
                verify_checkpoint(self.out,state,sha(ROOT/self.args.config))
                locked[arm]=state['best']
            rank=lambda r:(-r['successes'],r['time'],r['control'],r['arm'])
            winner=min((locked[a] for a in ('A1','A2','A3')),key=rank)['arm']
            lock=dict(arms=locked,ae_winner=winner,reader_hash=sha(self.out/'reader.pth'))
            write_json(self.out/'selection.json',lock)
            for arm in ARMS:
                path=self.out/'confirmation'/f'{arm}.json'
                if path.exists():continue
                b=locked[arm]
                result=self.evaluate(arm,torch.tensor(b['motion'],device=self.device),
                    torch.tensor(b['camera'],device=self.device),'confirmation',record=True)
                write_json(path,result);print('CONFIRM',arm,result['successes'],'/64',flush=True)
            a=locked[winner]
            for joint in ('C','D'):
                j=locked[joint]
                for name,m,c,mode in [
                    ('camera',a['motion'],j['camera'],'learned'),
                    ('motion',j['motion'],a['camera'],MODES[winner])]:
                    tag=f'{joint}_cross_{name}';path=self.out/'confirmation'/f'{tag}.json'
                    if not path.exists():
                        result=self.evaluate(joint,torch.tensor(m,device=self.device),torch.tensor(c,device=self.device),
                            'confirmation',record=True,mode=mode,tag=tag)
                        write_json(path,result)
                # Developer-defined example is id0, regardless of its outcome.
                for tag,m,c,mode in [('joint',j['motion'],j['camera'],'learned'),
                    ('cross_camera',a['motion'],j['camera'],'learned'),
                    ('cross_motion',j['motion'],a['camera'],MODES[winner]),
                    ('ae',a['motion'],a['camera'],MODES[winner])]:
                    _,ep,tr,_=self.run('development',[0],torch.tensor(m,device=self.device),
                        torch.tensor(c,device=self.device),mode,phase='representative',arm=joint,record=True)
                    save(self.out/'examples'/f'{joint}_{tag}.pth',dict(episodes=ep,timeline=tr))
            write_json(self.out/'evaluated.json',dict(complete=True))

    def report(self):
        from bounded_route.report import make_report
        make_report(self.out)


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--stage',choices=('prepare','train','evaluate','report','all'),required=True)
    parser.add_argument('--out_dir',required=True)
    parser.add_argument('--config',default='configs/gray_bounded_joint.args')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--rounds',type=int)
    parser.add_argument('--gradient-microbatch',type=int,choices=(1,2,4),default=4)
    args=parser.parse_args()
    os.chdir(ROOT)
    exp=Experiment(args)
    for stage in ('prepare','train','evaluate','report') if args.stage=='all' else (args.stage,):
        print('STAGE',stage,flush=True);getattr(exp,stage if stage!='evaluate' else 'evaluate_all')()


if __name__=='__main__':main()
