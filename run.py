import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .core import network,modes,demonstrations,fit,certificate,rollout

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='results');p.add_argument('--seed',type=int,default=7);p.add_argument('--buses',type=int,default=3);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    plants=modes(network(a.buses),[.2,.5,1.,2.,5.,9.]);X,U=demonstrations(plants,seed=a.seed)
    K,P,opt=fit(plants,X,U)
    baseline=np.linalg.solve(X@X.T+1e-6*np.eye(X.shape[0]),X@U.T).T
    seq=np.random.default_rng(a.seed+1000).integers(len(plants),size=500)
    x0=np.r_[np.zeros(a.buses),np.full(a.buses,.05)]
    report=dict(seed=a.seed,buses=a.buses,dt=.02,synthetic_network=True,optimization=opt,controllers={})
    fig,ax=plt.subplots(figsize=(8,4))
    for name,gain in [('certified_fixed_P',K),('unconstrained_ridge',baseline)]:
        states,inputs=rollout(plants,gain,x0,seq,.02);omega=states[:,a.buses:]
        report['controllers'][name]=dict(**certificate(plants,gain,P),imitation_mse=float(np.mean((gain@X-U)**2)),sampled_frequency_l1=float(.02*np.abs(omega[:-1]).sum()),sampled_control_l1=float(.02*np.abs(inputs).sum()),final_state_norm=float(np.linalg.norm(states[-1])))
        np.savez(out/(name+'.npz'),K=gain,P=P,states=states,inputs=inputs,sequence=seq)
        header='time,'+','.join([f'theta_{i}' for i in range(a.buses)]+[f'omega_{i}' for i in range(a.buses)])
        np.savetxt(out/(name+'.csv'),np.column_stack([np.arange(len(states))*.02,states]),delimiter=',',header=header,comments='')
        ax.plot(np.arange(len(states))*.02,omega[:,0],label=name)
    ax.set(xlabel='Time (model units)',ylabel='Bus 0 frequency deviation (model units)');ax.legend();fig.tight_layout();fig.savefig(out/'frequency_response.png',dpi=160);plt.close(fig)
    (out/'metrics.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
