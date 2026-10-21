"""Check actual hard-forward losses after matched first optimizer updates."""
import copy,json,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import torch
from config import set_global_seed
from tools.intent_common import load_intent_args
from intent_model import build_policy
from train_utils import build_env
from trainer import _rollout,_loss_from_rollout,_compute_success_collision
from flight_replay import flight_sequence,masked_action_anchor

torch.set_num_threads(1)
args=load_intent_args('configs/gray_joint_parallax_replay.args',batch_size=8)
device=torch.device('cpu')
template=build_policy(args);template.load_state_dict(torch.load(args.resume,map_location='cpu'));template.eval()
data=torch.load(args.flight_replay_dataset,map_location='cpu')['data'];ids=torch.tensor([1,11,21,31])
with torch.no_grad():target=flight_sequence(template,data,ids,device)
valid=data['valid'][ids]
def objective(model,seed,mode,grad):
 args.observation_grad_mode=args.sensor_grad_mode=mode
 set_global_seed(seed);env=build_env(args.batch_size,args,device);env.reset('nominal')
 with torch.set_grad_enabled(grad):
  rollout=_rollout(env,model,args,args.batch_size,device,False,None,False)
  nav,terms,p,clearance=_loss_from_rollout(rollout,env,args)
  replay=masked_action_anchor(flight_sequence(model,data,ids,device),target,valid)
  loss=nav+args.flight_replay_weight*replay
  success,collision,distance=_compute_success_collision(p,clearance,env,args,rollout)
 result={'total_loss':float(loss),'navigation_loss':float(nav),'replay_contribution':float(replay)*args.flight_replay_weight,'success':float(success),'collision':float(collision),'goal_distance':float(distance)}
 return loss,result
result={'config':'configs/gray_joint_parallax_replay.args','batch':8,'train_seed':789301,'holdout_seed':789302,'optimizer':'actual first AdamW update after clipping to norm 5','limitation':'Single update with discontinuous hard-forward objectives; seed pairs are local development evidence, not evidence of trained-policy superiority.','conditions':{}}
_,result['before_train']=objective(template,789301,'full',False)
_,result['before_holdout']=objective(template,789302,'full',False)
gradients={}
for mode in ('full','detached'):
 model=copy.deepcopy(template)
 loss,initial=objective(model,789301,mode,True);loss.backward()
 gradients[mode]={n:torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for n,p in model.named_parameters()}
 del model
for mode in ('full','detached'):
 rows=[]
 for lr in (1e-5,3e-5,1e-4):
  model=copy.deepcopy(template);optim=torch.optim.AdamW(model.parameters(),lr=lr)
  for n,p in model.named_parameters():p.grad=gradients[mode][n].clone()
  norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),5.))
  optim.step()
  _,train=objective(model,789301,'full',False);_,holdout=objective(model,789302,'full',False)
  row={'lr':lr,'gradient_norm':norm,'train':train,'holdout':holdout}
  rows.append(row);print(mode,row,flush=True)
 result['conditions'][mode]=rows
Path('logs/sparse_poles_iteration_20261021/parallax_update_direction_audit.json').write_text(json.dumps(result,indent=2)+'\n')
