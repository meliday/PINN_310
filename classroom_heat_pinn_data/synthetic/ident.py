import numpy as np
from solver import solve, sample
LAYOUT_A=np.array([(0.3,0.04),(0.7,0.04),(0.95,0.25),(0.95,0.55),(0.3,0.76),(0.7,0.76),(0.3,0.4),(0.5,0.25),(0.7,0.4),(0.5,0.55)])
def run(Bi,heaters,thc=0.9,nx=101):
    return solve(nx,{"S":Bi[0],"E":Bi[1],"N":Bi[2]},{"N":thc},heaters=heaters)
def ident(Bi,heaters,pts=LAYOUT_A,sigC=0.1,dT=20,thc=0.9):
    Bi=np.array(Bi,float); J=np.zeros((len(pts),3))
    for p in range(3):
        d=np.zeros(3); d[p]=1e-4*Bi[p]
        up=run(Bi+d,heaters,thc); dn=run(Bi-d,heaters,thc)
        J[:,p]=(sample(up[0],up[1],up[2],pts)-sample(dn[0],dn[1],dn[2],pts))/(2*d[p])
    Jr=J*Bi; C=np.linalg.inv(Jr.T@Jr); sd=np.sqrt(np.diag(C))*sigC/dT
    return sd, C[0,2]/np.sqrt(C[0,0]*C[2,2])
