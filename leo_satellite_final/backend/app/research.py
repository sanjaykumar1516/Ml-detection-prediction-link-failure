"""Reproducible synthetic cascade benchmark and learned routing baseline."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from functools import lru_cache
from app.simulator import snapshot
import networkx as nx

EDGES=[(i,j) for i in range(12) for j in range(i+1,12)]
IDS=[f'SAT-{i+1:02d}|SAT-{j+1:02d}' for i,j in EDGES]
A=np.array([[int(a!=b and bool(set(a)&set(b))) for b in EDGES] for a in EDGES],float)
A/=A.sum(axis=1,keepdims=True)

def episode(seed, steps=100):
    rng=np.random.default_rng(seed); load=rng.uniform(.25,.65,66); capacity=rng.uniform(.75,1.25,66); failed=np.zeros(66,bool); rows=[]
    trigger=int(rng.integers(15,30)); edge=int(rng.integers(66)); cause=['hardware','interference','overload','handover'][seed%4]
    for t in range(steps):
        old=failed.copy()
        if t==trigger: failed[edge]=True
        load=np.clip(load+rng.normal(0,.025,66)+A@(old.astype(float)*.055),0,2)
        failed |= (load>capacity)&(rng.random(66)<.35)
        rows.append(np.stack([load,capacity,failed.astype(float)],axis=1))
    return np.array(rows),cause

def features(x,t):
    z=x[t]; return np.concatenate([z,A@z,z-x[max(0,t-5)],A@(z-x[max(0,t-5)])],axis=1)

def dataset(seeds):
    for seed in seeds:
        x,cause=episode(seed)
        for t in range(5,len(x)-15):
            mask=x[t,:,2]==0; y=x[t+15,:,2].astype(int)
            yield features(x,t)[mask],y[mask]

@lru_cache(maxsize=1)
def train():
    import torch
    from torch import nn
    torch.manual_seed(42); torch.set_num_threads(1)
    class GNN(nn.Module):
        def __init__(self):
            super().__init__(); self.local=nn.Linear(6,16); self.out=nn.Linear(32,1)
        def forward(self,x):
            h=torch.relu(self.local(torch.cat([x[:,:,:3],x[:,:,6:9]],dim=-1))); return self.out(torch.cat([h,torch.tensor(A,dtype=torch.float32)@h],dim=-1)).squeeze(-1)
    def batch(seeds):
        xx=[]; yy=[]; mm=[]
        for seed in seeds:
            x,_=episode(seed)
            for t in range(5,85,3):
                xx.append(features(x,t)); yy.append(x[t+15,:,2]); mm.append(x[t,:,2]==0)
        return torch.tensor(np.array(xx),dtype=torch.float32),torch.tensor(np.array(yy),dtype=torch.float32),torch.tensor(np.array(mm))
    x,y,m=batch(range(24)); model=GNN(); opt=torch.optim.Adam(model.parameters(),lr=.015)
    for _ in range(60):
        opt.zero_grad(); logits=model(x); loss=nn.functional.binary_cross_entropy_with_logits(logits[m],y[m]); loss.backward(); opt.step()
    v,w,k=batch(range(100,108)); cal,cy,cm=batch(range(200,208))
    with torch.no_grad(): p=torch.sigmoid(model(v))[k].numpy(); cp=torch.sigmoid(model(cal))[cm].numpy()
    truth=w[k].numpy(); baseline=v[:,:,0][k].numpy()/v[:,:,1][k].numpy(); q=float(np.quantile(abs(cy[cm].numpy()-cp),.9,method='higher'))
    return model,{'gnn_pr_auc':float(average_precision_score(truth,p)), 'gnn_f1':float(f1_score(truth,p>=.5,zero_division=0)), 'overload_pr_auc':float(average_precision_score(truth,baseline)), 'test_examples':len(p),'calibration_residual_q90':q}

def report(seed=500):
    import torch
    model,metrics=train(); x,cause=episode(seed); t=35
    with torch.no_grad(): p=torch.sigmoid(model(torch.tensor(features(x,t)[None],dtype=torch.float32)))[0].numpy()
    q=metrics['calibration_residual_q90']; ranked=sorted([{'link':IDS[i],'risk':float(p[i]),'upper_score':min(1,float(p[i])+q),'failed_next_15s':bool(x[t+15,i,2])} for i in range(66) if not x[t,i,2]],key=lambda r:-r['risk'])
    return {'metrics':metrics,'scenario_seed':seed,'time_s':t,'links':ranked,'note':'Separate synthetic load cascade experiment; temporal differences + graph message passing. Residual upper score is heuristic, not a calibrated probability bound.'}

def benchmark(seed=500):
    x,cause=episode(seed)
    return {'seed':seed,'scenario':cause,'records':[{'time_s':t,'link':IDS[i],'load':float(x[t,i,0]),'capacity':float(x[t,i,1]),'failed':bool(x[t,i,2]),**{f'failure_within_{h}s':int(not x[t,i,2] and x[t+h,i,2]) if t+h<len(x) else None for h in (15,30,60)}} for t in range(len(x)) for i in range(66)]}

def routing_trial(seed=500):
    """Tabular Q-learning over candidate paths; separate training/test episodes."""
    def run(seed,policy,learn=False):
        x,_=episode(seed); total=0; outages=0; previous=None
        for t in range(5,85):
            g=nx.Graph(); g.add_nodes_from(range(12))
            for i,(a,b) in enumerate(EDGES):
                if not x[t,i,2]: g.add_edge(a,b,weight=1+float(x[t,i,0]))
            try: paths=list(__import__('itertools').islice(nx.shortest_simple_paths(g,0,6,weight='weight'),3))
            except nx.NetworkXNoPath: paths=[]
            if not paths: outages+=1; continue
            indices=[[EDGES.index(tuple(sorted(e))) for e in zip(p,p[1:])] for p in paths]
            scores=[max(x[t,i,0]/x[t,i,1]+max(0,x[t,i,0]-x[t-5,i,0]) for i in ix) for ix in indices]
            state=tuple(min(3,int(s*2)) for s in scores)+(len(paths),)
            values=Q.setdefault(state,np.zeros(3))
            action=int(rng.integers(len(paths))) if learn and rng.random()<.2 else int(np.argmax(values[:len(paths)])) if policy=='learned' else 0
            ix=indices[action]; delivered=not any(x[t+1,i,2] for i in ix); reward=float(delivered)-.1*(previous is not None and previous!=paths[action]); previous=paths[action]
            if learn: values[action]+=.15*(reward-values[action])
            total+=int(delivered); outages+=int(not delivered)
        return {'delivered_steps':total,'outage_steps':outages}
    Q={}; rng=np.random.default_rng(42)
    for s in range(40): run(s,'learned',True)
    return {'reactive':run(seed,'reactive'),'learned':run(seed,'learned'),'note':'Contextual bandit routing baseline (immediate reward), not PPO or conformal RL; separate synthetic cascade environment.'}
