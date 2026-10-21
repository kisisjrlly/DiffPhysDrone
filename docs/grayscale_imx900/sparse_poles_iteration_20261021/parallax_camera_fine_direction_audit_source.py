"""Local constant-command sensitivity; not a learned camera controller."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import torch
from config import set_global_seed
from tools.intent_common import load_intent_args
from intent_model import build_policy
from train_utils import build_env
from trainer import _rollout,_loss_from_rollout,_compute_success_collision

torch.set_num_threads(1)
args=load_intent_args('configs/gray_joint_parallax_lookahead.args',batch_size=8,scenarios=['nominal','dark','bright'])
actor=build_policy(args).eval();actor.load_state_dict(torch.load('logs/sparse_poles_iteration_20261021/parallax_lookahead_flight_imitation_v1/final.pth',map_location='cpu'));actor.requires_grad_(False)
class FixedCommand(torch.nn.Module):
 def __init__(self,eg):super().__init__();self.eg=eg
 def forward(self,*v,**kw):
  act,_,h,ch=actor(*v,**kw)
  return act,self.eg[None].expand(args.batch_size,-1),h,ch

def run(scene,e,g,grad):
 set_global_seed(791101);env=build_env(args.batch_size,args,torch.device('cpu'));env.reset(scene)
 eg=torch.tensor([e,g],requires_grad=grad)
 with torch.set_grad_enabled(grad):
  ro=_rollout(env,FixedCommand(eg),args,args.batch_size,torch.device('cpu'),False,None,False)
  loss,terms,p,clear=_loss_from_rollout(ro,env,args)
  sg,cl,dist=_compute_success_collision(p,clear,env,args,ro)
  derivative=torch.autograd.grad(loss,eg)[0].tolist() if grad else None
 row={'command_E':e,'command_G':g,'loss':float(loss),'success':float(sg),'collision':float(cl),'goal_distance':float(dist),'derivative_E_G':derivative}
 print(scene,row,flush=True);return row
result={'seed':791101,'batch':8,'meaning':'Fine local changes with common random numbers; terminal/raster discontinuities may remain.','scenarios':{}}
for scene in args.scenarios:
 base=run(scene,.08,.05,True)
 rows={}
 for eps in (1e-4,1e-3):
  minus=run(scene,.08-eps,.05,False);plus=run(scene,.08+eps,.05,False)
  rows[str(eps)]={'minus':minus,'plus':plus,'central_difference':(plus['loss']-minus['loss'])/(2*eps)}
 result['scenarios'][scene]={'base':base,'differences':rows}
Path('logs/sparse_poles_iteration_20261021/parallax_camera_fine_direction_audit.json').write_text(json.dumps(result,indent=2)+'\n')
