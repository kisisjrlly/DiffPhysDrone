"""Physical externally imposed warmup diagnostics, never learned motion."""
import json,sys,hashlib
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import torch
from config import set_global_seed
from tools.intent_common import load_intent_args
from tools.evaluate_pole_coordination import initial_probe_override
from intent_model import build_policy
from train_utils import build_env
from eval import run_one_episode

torch.set_num_threads(1)
root=Path("logs/sparse_poles_iteration_20261021")
checkpoint=root/"parallax_lookahead_flight_imitation_v1/final.pth"
args=load_intent_args("configs/gray_joint_parallax_lookahead.args",batch_size=1,camera_control_mode="fixed")
actor=build_policy(args).eval();actor.load_state_dict(torch.load(checkpoint,map_location="cpu"))
result={"seed_range":[790000,790039],"checkpoint_sha256":hashlib.sha256(checkpoint.read_bytes()).hexdigest(),"device":"cpu","claim":"External first-24-step intervention diagnostics only, no autonomous or gradient-learning claim.","conditions":{}}
for condition,amplitude in [("hover_first24",0.),("lateral_probe_first24",.12)]:
 env=build_env(1,args,torch.device("cpu"),eval_mode=True);rows=[]
 with torch.no_grad():
  for i in range(40):
   row,_=run_one_episode(i,"nominal",args,actor,env,None,torch.device("cpu"),episode_seed=790000+i,action_override=initial_probe_override(args,amplitude if i%2 else -amplitude))
   row["near_pole_side"]=int(torch.sign(env.cyl[0,0,1]));rows.append(row)
 result["conditions"][condition]={"successes":sum(r["success_rate"] for r in rows),"n":40,"episodes":rows}
 (root/"parallax_probe_vs_wait.json").write_text(json.dumps(result,indent=2)+"\n")
 print(condition,result["conditions"][condition]["successes"],flush=True)
