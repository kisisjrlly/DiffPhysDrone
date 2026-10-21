import json,sys,hashlib
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import torch
from tools.intent_common import load_intent_args
from config import set_global_seed
from intent_model import build_policy
from train_utils import build_env
from trainer import _rollout,_loss_from_rollout,_compute_success_collision

torch.set_num_threads(1)
args=load_intent_args('configs/gray_joint_parallax_lookahead.args',batch_size=16)
checkpoint=Path('logs/sparse_poles_iteration_20261021/parallax_lookahead_flight_imitation_v1/final.pth')
template=build_policy(args).eval();template.load_state_dict(torch.load(checkpoint,map_location='cpu'));template.requires_grad_(False)
class Perturb(torch.nn.Module):
    uses_camera_timeline=True
    def __init__(self,delta):super().__init__();self.actor=template;self.delta=delta;self.t=0
    def forward(self,*v,**kw):
        noise=self.delta*torch.tensor([0.,1.,0.])[None].expand(args.batch_size,-1)*(self.t<12)
        self.t+=1
        return self.actor(*v,**kw,action_noise=noise)
def run(value,mode,grad):
    args.observation_grad_mode=args.sensor_grad_mode=mode
    set_global_seed(787301);env=build_env(args.batch_size,args,torch.device('cpu'));env.reset('nominal')
    env.cyl[8:]=env.cyl[:8].clone();env.cyl[8:,:,1]*=-1
    env.gray_light_factor[8:]=env.gray_light_factor[:8].clone()
    delta=torch.tensor(float(value),requires_grad=grad)
    with torch.set_grad_enabled(grad):
        rollout=_rollout(env,Perturb(delta),args,args.batch_size,torch.device('cpu'),False,None,False)
        loss,_,p,clearance=_loss_from_rollout(rollout,env,args)
        success,collision,distance=_compute_success_collision(p,clearance,env,args,rollout)
        derivative=float(torch.autograd.grad(loss,delta)[0]) if grad else None
    out={'lateral_acceleration_offset':value,'loss':float(loss),'success':float(success),'collision':float(collision),'goal_distance':float(distance),'autograd':derivative}
    print(mode,out,flush=True);return out
result={'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'batch':16,'seed':787301,'prefix_steps':12,'prefix_seconds':12/15,'config':'configs/gray_joint_parallax_lookahead.args','gradient_decay':args.grad_decay,'limitation':'Frozen actor, diagnostic action perturbation only. Masked terminal events, detached controller targets and decayed dynamics gradients mean the autograd estimator is approximate, not an exact derivative of this hard stochastic rollout. No claim of learned sensing.'}
result['gradients']={mode:run(0.,mode,True) for mode in ('full','detached')}
assert result['gradients']['full']['loss']==result['gradients']['detached']['loss']
result['finite_difference']={}
for eps in (.1,.3):
    minus=run(-eps,'full',False);plus=run(eps,'full',False)
    result['finite_difference'][str(eps)]={'minus':minus,'plus':plus,'central_slope':(plus['loss']-minus['loss'])/(2*eps)}
Path('logs/sparse_poles_iteration_20261021/parallax_early_action_gradient_audit.json').write_text(json.dumps(result,indent=2)+'\n')
