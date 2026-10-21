"""Reconstruct CUDA reset layouts for retained paired episodes; no new training."""
import json,sys,hashlib
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import torch
from config import build_parser,parse_scenarios,validate_args,set_global_seed
from train_utils import build_env
base=Path("logs/sparse_poles_iteration_20261021")
result={"device":"cuda","interpretation":"Layout-conditional diagnostics on existing autonomous trajectories; no external action override.","runs":{}}
for run in ["parallax_replay_small_step_joint_pair_v1","parallax_consistent_joint_pair_v1"]:
 manifest=json.loads((base/run/"pilot_manifest.json").read_text());out={}
 args=build_parser().parse_args([]);vars(args).update(json.loads((base/run/"joint/args.json").read_text()))
 args.batch_size=1;args.scenarios=parse_scenarios(args.scenarios);args.vis_enable=False
 env=build_env(1,args,torch.device("cuda"),eval_mode=True)
 signs=[];layouts=[]
 for seed in manifest["evaluation_episode_seeds"]:
  set_global_seed(seed,args.deterministic);env.reset("nominal")
  signs.append(int(torch.sign(env.cyl[0,0,1])));layouts.append(env.cyl[0].cpu().tolist())
 for method in ["joint","nondiff_sensing"]:
  rows=json.loads((base/run/method/"episodes.json").read_text());traces=json.loads((base/run/method/"traces.json").read_text())
  assert [r["episode_seed"] for r in rows]==manifest["evaluation_episode_seeds"]
  groups={}
  for r in traces:groups.setdefault(r["episode_idx"],[]).append(r)
  out[method]={"checkpoint_sha256":manifest["methods"][method]["checkpoint_sha256"],"by_near_pole_side":{}}
  for sign in [-1,1]:
   ids=[i for i,s in enumerate(signs) if s==sign]
   out[method]["by_near_pole_side"][str(sign)]={"n":len(ids),"successes":sum(rows[i]["success_rate"] for i in ids),"first24_y":[next((r for r in groups[i] if r["step"]==23),groups[i][-1])["local_y"] for i in ids]}
 result["runs"][run]={"episode_seeds":manifest["evaluation_episode_seeds"],"near_pole_sides":signs,"layouts":layouts,"methods":out}
 print(run,{k:{s:{kk:vv for kk,vv in val.items() if kk!="first24_y"} for s,val in v["by_near_pole_side"].items()} for k,v in out.items()},flush=True)
(base/"parallax_layout_outcomes.json").write_text(json.dumps(result,indent=2)+"\n")
