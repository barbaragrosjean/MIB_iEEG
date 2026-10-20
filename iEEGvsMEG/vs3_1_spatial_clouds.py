"""Unpaired native-feature cloud comparisons; no electrode/source matching.

All features contribute to weighted training prototypes (default 32). Reported
OT/GW/MMD are prototype approximations, not exact full-cloud distances. Equal
feature mass is used; this is not participant-balanced. No anatomical region aggregation is performed.
"""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linprog
from scipy.sparse import kron, eye, csr_matrix, vstack
from scipy.linalg import orthogonal_procrustes
from sklearn.cluster import KMeans


def transport(cost,a,b):
    n,m=cost.shape
    constraints=vstack([kron(eye(n),np.ones((1,m))),kron(np.ones((1,n)),eye(m))],format='csr')
    fit=linprog(cost.ravel(),A_eq=constraints,b_eq=np.r_[a,b],bounds=(0,None),method='highs')
    if not fit.success:raise RuntimeError(f'OT failed: {fit.message}')
    return fit.x.reshape(n,m)


def gw_loss(cx,cy,a,b,initial,iterations=100):
    """Squared internal-distance loss, conditional gradient, local solution.

    Objective: https://pythonot.github.io/_modules/ot/gromov/_gw.html
    Returns the squared-loss objective, not a globally certified GW distance.
    """
    const=float(a@(cx*cx)@a+b@(cy*cy)@b)
    def loss(t):return const-2*np.sum((cx@t@cy.T)*t)
    t=initial.copy();converged=False
    for iteration in range(iterations):
        gradient=-4*cx@t@cy.T
        direction=transport(gradient,a,b)-t
        gap=-float(np.sum(gradient*direction))
        if gap<1e-9:converged=True;break
        # Exact quadratic line search over the feasible segment.
        f0=loss(t);f1=loss(t+direction);fh=loss(t+.5*direction)
        q=2*(f1+f0-2*fh);linear=f1-f0-q
        candidates=[0.,1.]
        if q>0:candidates.append(float(np.clip(-linear/(2*q),0,1)))
        step=min(candidates,key=lambda z:loss(t+z*direction))
        t+=step*direction
    return max(0.,loss(t)),converged,iteration+1


def distances(x,y,a,b,bandwidth):
    squared=cdist(x,y,'sqeuclidean');plan=transport(squared,a,b)
    w2=float(np.sum(plan*squared))
    cx=cdist(x,x);cy=cdist(y,y)
    # Two feasible starts; GW is nonconvex and not certified globally optimal.
    candidates=[gw_loss(cx,cy,a,b,start) for start in (np.outer(a,b),plan)]
    gw,converged,it=min(candidates,key=lambda v:v[0])
    kernel=lambda u,v:np.exp(-cdist(u,v,'sqeuclidean')/(2*bandwidth**2))
    mmd=float(a@kernel(x,x)@a+b@kernel(y,y)@b-2*a@kernel(x,y)@b)
    return dict(wasserstein2=np.sqrt(max(0,w2)),gw_squared_loss=gw,
                mmd2=max(0,mmd),gw_converged=converged,gw_iterations=it)


def evaluate_clouds(patterns, scores, *, prototypes=32, seed=2026):
    """Training temporal Procrustes aligns axes; training spatial normalization.

    Prototypes fit training native features only. Fixed training memberships
    aggregate corresponding test feature rows; test counts never select k.
    """
    if prototypes<2:raise ValueError('At least two prototypes required.')
    ti=scores['train']['ieeg'];tm=scores['train']['meg']
    rotation,_=orthogonal_procrustes(ti-ti.mean(0),tm-tm.mean(0))
    native={};cloud={};weights={};saved={'temporal_rotation':rotation}
    for m in ('ieeg','meg'):
        parts={p:(x@rotation if m=='ieeg' else x) for p,x in patterns[m].items()}
        center=parts['train'].mean(0);scale=np.sqrt(np.mean(np.sum((parts['train']-center)**2,axis=1)))
        if scale<=np.finfo(float).eps:raise ValueError('Spatial cloud has zero training spread.')
        native[m]={p:(x-center)/scale for p,x in parts.items()}
        train=native[m]['train'];n=len(train)
        if n<=prototypes:labels=np.arange(n)
        else:labels=KMeans(n_clusters=min(prototypes,len(np.unique(train,axis=0))),n_init=5,random_state=seed).fit_predict(train)
        ids=np.unique(labels);weights[m]=np.array([(labels==i).sum()/n for i in ids])
        cloud[m]={p:np.stack([x[labels==i].mean(0) for i in ids]) for p,x in native[m].items()}
        saved[m+'_membership']=labels;saved[m+'_mass']=weights[m]
        saved[m+'_center']=center;saved[m+'_scale']=scale
        for p,x in cloud[m].items():saved[m+'_'+p+'_prototypes']=x
    pooled=np.vstack([cloud[m]['train'] for m in ('ieeg','meg')]);d=cdist(pooled,pooled)
    positive=d[d>0]; bandwidth=float(np.median(positive)) if len(positive) else 1.
    saved['mmd_bandwidth']=bandwidth
    rows=[]
    for part in scores:
        rows.append(dict(partition=part,**distances(cloud['ieeg'][part],cloud['meg'][part],weights['ieeg'],weights['meg'],bandwidth),
                         ieeg_features=len(native['ieeg'][part]),meg_features=len(native['meg'][part]),
                         ieeg_prototypes=len(weights['ieeg']),meg_prototypes=len(weights['meg']),mmd_bandwidth=bandwidth))
    return rows,saved
