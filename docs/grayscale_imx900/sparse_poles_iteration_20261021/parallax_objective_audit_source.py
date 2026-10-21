import json, sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
import torch
from config import set_global_seed
from tools.intent_common import load_intent_args
from tools.pole_navigation_teacher import PoleNavigationTeacher,bounded_probe_command
from intent_model import build_policy
from trainer import _rollout,_loss_from_rollout,_compute_success_collision,_build_loss_contrib_metrics
from train_utils import build_env

torch.set_num_threads(4)
args=load_intent_args('configs/gray_joint_parallax.args', batch_size=32)
class Diagnostic(torch.nn.Module):
    def __init__(self, env, name):
        super().__init__(); self.env=env; self.name=name; self.step=0
        self.teacher=PoleNavigationTeacher(env,args,clearance_margin=.06) if name=='probe_then_teacher' else None
    def forward(self,state,hx=None,**kw):
        if self.name=='probe_then_teacher':
            if self.step<24:
                act,_=bounded_probe_command(self.env,args,self.step,state.new_full((args.batch_size,),.12))
            else: act,_=self.teacher.command(self.env,(self.step-24)/args.base_control_freq)
        elif self.name=='hover': act=state.new_zeros(args.batch_size,3)
        else:
            act=state.new_zeros(args.batch_size,3); act[:,0]=args.max_acc_cmd
        self.step+=1
        return act,state.new_tensor([.08,.05])[None].expand(args.batch_size,-1),hx,None
result={}
for name in ('probe_then_teacher','hover','straight','common_actor','trained_actor'):
    set_global_seed(784301)
    env=build_env(args.batch_size,args,torch.device('cpu')); env.reset('nominal')
    if name in ('common_actor','trained_actor'):
        model=build_policy(args)
        ckpt='parallax_post_measurement_imitation_v1/final.pth' if name=='common_actor' else 'parallax_joint_pair_post_measurement_v1/joint/final.pth'
        model.load_state_dict(torch.load('logs/sparse_poles_iteration_20261021/'+ckpt,map_location='cpu')); model.eval()
    else: model=Diagnostic(env,name)
    with torch.no_grad():
        rollout=_rollout(env,model,args,args.batch_size,torch.device('cpu'),False,None,False)
        loss,terms,p,clearance=_loss_from_rollout(rollout,env,args)
        success,collision,distance=_compute_success_collision(p,clearance,env,args,rollout)
    result[name]={'loss':float(loss),'success':float(success),'collision':float(collision),'goal_distance':float(distance),'contributions':_build_loss_contrib_metrics(terms,args)}
    print(name,result[name],flush=True)
Path('logs/sparse_poles_iteration_20261021/parallax_objective_audit.json').write_text(json.dumps(result,indent=2)+'\n')
