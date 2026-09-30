import ast, math, os, tempfile
from pathlib import Path
import torch
root = Path(__file__).resolve().parents[1] / "dc-sae"

def functions(path, names):
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    ns = dict(torch=torch, math=math, os=os)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)
    return ns

ev = functions(root/'eval_gfid.py', {'sample_latent','resolve_eval_time_shift','load_dit_checkpoint'})
tr = functions(root/'train_dit.py', {'sample_latent'})
resolve = ev['resolve_eval_time_shift']
for flag in ['zero_hf_mode','hf_zero_then_joint_mode']:
    assert resolve({flag: True}, (1024,16,16),768) == math.sqrt(48)
    assert resolve({flag: True}, (832,8,8),768) == math.sqrt(12)
assert resolve({}, (1024,16,16),768) == 8.0
assert resolve({'time_dist_shift_base':1024}, (1024,16,16),768) == 16.0
class Toy(torch.nn.Module):
    def forward(self,x,t,y):
        return .4*torch.tanh(x) + t[:,None,None,None]**2
args=dict(model=Toy(), batch_size=2, latent_shape=(4,3,3),device=torch.device('cpu'),y=torch.tensor([1,2]),steps=50,use_cfg=False)
torch.manual_seed(42)
a=tr['sample_latent'](**args,time_shift=math.sqrt(48))
torch.manual_seed(42)
b=ev['sample_latent'](**args,time_shift=resolve({'hf_zero_then_joint_mode':True},(1024,16,16),768))
assert torch.equal(a,b)
torch.manual_seed(42)
c=ev['sample_latent'](**args,time_shift=8)
assert not torch.equal(a,c)
with tempfile.TemporaryDirectory() as d:
    m=torch.nn.Linear(2,1,bias=False)
    p=os.path.join(d,'weights.pt')
    torch.save({'ema':{'weight':torch.ones(1,2)},'model':{'weight':torch.zeros(1,2)},'train_steps':400000},p)
    assert ev['load_dit_checkpoint'](m,p,torch.device('cpu')) == 400000
    assert m._eval_weight_source == 'ema' and torch.equal(m.weight,torch.ones(1,2))
print('PASS: phase shifts; full-channel default; shift base; identical no-CFG sampling; old shift changes output; EMA priority and metadata')
