import numpy as np
import pytest
from frequency_control.core import network,modes,discretize,demonstrations,feasible_seed,certificate,fit,rollout

def test_network():
    L=network();assert np.allclose(L,L.T) and np.allclose(L.sum(axis=1),0)
def test_seed():
    plants=modes(network(),[.2,1,9,np.array([.3,2,5])]);K,P=feasible_seed(plants);assert certificate(plants,K,P)['certified']
def test_discretize():
    A,B=discretize(np.zeros((2,2)),np.eye(2),.1);assert np.allclose(A,np.eye(2)) and np.allclose(B,.1*np.eye(2))
def test_demonstrations():
    plants=modes(network(),[.5,2]);X,U=demonstrations(plants,horizon=8,scenarios=2);X2,U2=demonstrations(plants,horizon=8,scenarios=2)
    assert X.shape==(6,16) and U.shape==(3,16) and np.allclose(X,X2) and np.allclose(U,U2)
def test_fit():
    plants=modes(network(),[.5,2]);X,U=demonstrations(plants,horizon=12,scenarios=3);K,P,_=fit(plants,X,U,maxiter=80)
    assert certificate(plants,K,P)['certified']
    states,_=rollout(plants,K,np.ones(6)*.05,np.tile([0,1],500),.02);assert np.linalg.norm(states[-1])<np.linalg.norm(states[0])
def test_invalid_inertia():
    with pytest.raises(ValueError):modes(network(),[0])
