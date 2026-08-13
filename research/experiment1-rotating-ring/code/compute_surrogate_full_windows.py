#!/usr/bin/env python3
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(Path(__file__).resolve().parent))
from surrogate_model import SurrogateConfig, sample_clean_trajectories, add_ou_noise, exact_score_and_jacobian

cfg=SurrogateConfig()
rng=np.random.default_rng(7)
y,_=sample_clean_trajectories(rng,cfg,350)
indices=np.arange(cfg.T)
center=cfg.T//2
cache={}
rows=[]
for t in (0.05,0.20,0.70,1.50):
    x=add_ou_noise(rng,y,t)
    full,_,_=exact_score_and_jacobian(x,indices,t,cfg,cache)
    target=full[:,center]
    power=float(np.mean(np.sum(target**2,axis=1)))
    for L in range(center+1):
        lo=max(0,center-L); hi=min(cfg.T,center+L+1)
        subidx=np.arange(lo,hi)
        sub,_,_=exact_score_and_jacobian(x[:,lo:hi],subidx,t,cfg,cache)
        approx=sub[:,center-lo]
        rel_mse=float(np.mean(np.sum((approx-target)**2,axis=1))/max(power,1e-15))
        rows.append({'t':t,'window_radius':L,'frames':hi-lo,'relative_mse':rel_mse,'relative_rmse':rel_mse**0.5})
pd.DataFrame(rows).to_csv(ROOT/'data/surrogate/window_errors_full.csv',index=False)
print(pd.DataFrame(rows).tail())
