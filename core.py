"""Synthetic variable-inertia example; convention u=Kx."""
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize

def network(n=3):
    if n<3: raise ValueError('At least 3 buses required')
    W=np.zeros((n,n))
    for i in range(n): W[i,(i+1)%n]=W[(i+1)%n,i]=1
    return np.diag(W.sum(axis=1))-W

def modes(L,inertias,damping=1.):
    n=len(L); plants=[]
    for inertia in inertias:
        m=np.broadcast_to(np.asarray(inertia,dtype=float),(n,))
        if np.any(m<=0): raise ValueError('Positive inertia required')
        Mi=np.diag(1/m)
        plants.append((np.block([[np.zeros((n,n)),np.eye(n)],[-Mi@L,-damping*Mi]]),np.vstack([np.zeros((n,n)),Mi])))
    return plants

def discretize(A,B,dt):
    d,n=B.shape
    E=expm(np.block([[A,B],[np.zeros((n,d+n))]])*dt)
    return E[:d,:d],E[:d,d:]

def demonstrations(plants,dt=.02,horizon=60,scenarios=12,seed=7):
    rng=np.random.default_rng(seed); discrete=[discretize(A,B,dt) for A,B in plants]
    d,n=plants[0][1].shape; Q=np.diag([1.]*n+[100.]*n)*dt; R=np.eye(n)*dt
    xs=[];us=[]
    for _ in range(scenarios):
        seq=rng.integers(len(plants),size=(horizon+3)//4).repeat(4)[:horizon]
        S=np.zeros((d,d));gains=[]
        for q in seq[::-1]:
            A,B=discrete[q];K=-np.linalg.solve(R+B.T@S@B,B.T@S@A)
            gains.append(K);S=Q+A.T@S@A+A.T@S@B@K;S=(S+S.T)/2
        x=rng.normal(0,.1,d)
        for q,K in zip(seq,gains[::-1]):
            u=K@x;xs.append(x.copy());us.append(u.copy());A,B=discrete[q];x=A@x+B@u
    return np.array(xs).T,np.array(us).T

def certificate(plants,K,P):
    residual=[];abscissae=[]
    for A,B in plants:
        F=A+B@K;H=F.T@P+P@F
        residual.append(float(np.linalg.eigvalsh((H+H.T)/2).max()))
        abscissae.append(float(np.linalg.eigvals(F).real.max()))
    pmin=float(np.linalg.eigvalsh(P).min())
    return dict(p_min=pmin,worst_lyapunov_eigenvalue=max(residual),worst_spectral_abscissa=max(abscissae),certified=bool(pmin>1e-7 and max(residual)<-1e-7))

def feasible_seed(plants):
    n=plants[0][1].shape[1];I=np.eye(n);P=np.linalg.inv(np.block([[I,-I],[-I,2*I]]))
    for e in range(24):
        K=-2.**e*np.hstack([I,I])
        if certificate(plants,K,P)['certified']: return K,P
    raise RuntimeError('No numerically certified seed found')

def fit(plants,X,U,maxiter=150):
    K0,P=feasible_seed(plants);shape=K0.shape;N=X.shape[1]
    def obj(v):
        E=v.reshape(shape)@X-U
        return float(np.sum(E*E)/N+1e-6*np.dot(v,v))
    def jac(v):return (2*(v.reshape(shape)@X-U)@X.T/N+2e-6*v.reshape(shape)).ravel()
    def con(v):
        K=v.reshape(shape)
        return np.concatenate([np.linalg.eigvalsh(-((A+B@K).T@P+P@(A+B@K)))-1e-5 for A,B in plants])
    res=minimize(obj,K0.ravel(),jac=jac,method='SLSQP',constraints=[dict(type='ineq',fun=con)],options=dict(maxiter=maxiter,ftol=1e-9))
    candidate=res.x.reshape(shape);accepted=certificate(plants,candidate,P)['certified'];K=candidate if accepted else K0
    return K,P,dict(optimizer_success=bool(res.success),message=str(res.message),candidate_accepted=accepted,used_seed_fallback=not accepted,objective=obj(K.ravel()),initial_objective=obj(K0.ravel()))

def rollout(plants,K,x0,sequence,dt):
    E=[expm((A+B@K)*dt) for A,B in plants];x=np.array(x0,dtype=float);states=[x.copy()];inputs=[]
    for q in sequence:
        inputs.append(K@x);x=E[int(q)]@x;states.append(x.copy())
    return np.array(states),np.array(inputs)
