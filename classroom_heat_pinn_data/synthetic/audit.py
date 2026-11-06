import json, time, numpy as np
from solver import solve, sample
from ident import LAYOUT_A
kn=json.load(open('data/known.json')); TI=kn['theta_inf']; H=kn['heater']
BI=np.array([1.0,0.4,0.25])
def run(nx,Bi=BI,A=H['A'],k=None):
    h=dict(H,A=A); return solve(nx,{"S":Bi[0],"E":Bi[1],"N":Bi[2]},TI,heaters=[h],k=k)

print("(a) convergence order on the ACTUAL problem (Robin-Neumann corners, truncated Gaussian)")
sols={nx:run(nx)[2] for nx in (51,101,201,401)}
d1=np.abs(sols[51]-sols[101][::2,::2]).max(); d2=np.abs(sols[101]-sols[201][::2,::2]).max(); d3=np.abs(sols[201]-sols[401][::2,::2]).max()
print(f"   successive max diffs (degC): {20*d1:.2e} {20*d2:.2e} {20*d3:.2e}  -> orders {np.log2(d1/d2):.2f}, {np.log2(d2/d3):.2f}")
# where is the largest error? (corner singularity check)
e=np.abs(sols[201]-sols[401][::2,::2]); i,j=np.unravel_index(e.argmax(),e.shape); print(f"   largest 201-vs-401 diff at x={i/200:.3f}, y={j/200:.3f}")
print(f"   estimated error of 201x161 reference (Richardson, p=2): {20*d3/3:.2e} degC")

print("(b) Jacobian finite-difference step check")
def J(pts,eps,params='Bi'):
    out=[]
    base=np.r_[BI,H['A']]
    names=[0,1,2] if params=='Bi' else [0,1,2,3]
    for p in names:
        d=np.zeros(4); d[p]=eps*base[p]
        u=run(101,Bi=(base+d)[:3],A=(base+d)[3]); v=run(101,Bi=(base-d)[:3],A=(base-d)[3])
        out.append((sample(u[0],u[1],u[2],pts)-sample(v[0],v[1],v[2],pts))/(2*d[p])*base[p])
    return np.array(out).T
J1=J(LAYOUT_A,1e-4); J2=J(LAYOUT_A,1e-3)
print(f"   max rel diff between eps=1e-4 and 1e-3: {np.abs(J1-J2).max()/np.abs(J1).max():.1e}")
sv=np.linalg.svd(J1,compute_uv=False); print(f"   singular values (log-Bi sensitivities): {np.round(sv,4)}, condition number {sv[0]/sv[-1]:.1f}")

print("(c) identifiability if heater strength A is ALSO unknown (4 params)")
J4=J(LAYOUT_A,1e-4,params='all'); C4=np.linalg.inv(J4.T@J4); sd4=np.sqrt(np.diag(C4))*0.1/20
sv4=np.linalg.svd(J4,compute_uv=False)
print(f"   rel std @0.1C (Bi_S, Bi_E, Bi_N, A): {np.round(sd4*100,1)}%, condition number {sv4[0]/sv4[-1]:.1f}")
corr=C4/np.sqrt(np.outer(np.diag(C4),np.diag(C4))); print("   corr(A, Bi_S/E/N):",np.round(corr[3,:3],3))

print("(d) global uniqueness scan of the noise-free misfit (self-consistent 51x41 grid, layout A)")
t=time.time()
xs,ys,Tt,_=run(51); ix=np.rint(LAYOUT_A[:,0]*50).astype(int); iy=np.rint(LAYOUT_A[:,1]*50).astype(int); obs=Tt[ix,iy]
g=np.exp(np.linspace(np.log(0.05),np.log(10),14))
best=[]
for a in g:
  for b in g:
    for c in g:
      T=run(51,Bi=np.array([a,b,c]))[2]; best.append((np.sqrt(np.mean((T[ix,iy]-obs)**2))*20,a,b,c))
best.sort()
print(f"   {len(best)} grid points in Bi in [0.05,10]^3 ({time.time()-t:.0f}s). Lowest misfits (degC RMS):")
for r in best[:5]: print(f"     {r[0]:.4f}  Bi=({r[1]:.3f},{r[2]:.3f},{r[3]:.3f})")
# any distant point with small misfit?
far=[r for r in best if max(abs(np.log(r[1]/1.0)),abs(np.log(r[2]/0.4)),abs(np.log(r[3]/0.25)))>np.log(2)]
print(f"   smallest misfit among points >2x away from truth in some Bi: {far[0][0]:.3f} degC at Bi=({far[0][1]:.2f},{far[0][2]:.2f},{far[0][3]:.2f})")

print("(e) mismatch model: k on the walls")
LY=0.8; Kv=lambda X,Y: 1+0.25*np.sin(2*np.pi*X)*np.sin(np.pi*Y/LY)
s=np.linspace(0,1,11); print("   k on S,N,W,E walls max|k-1|:", max(np.abs(Kv(s,0*s)-1).max(),np.abs(Kv(s,LY+0*s)-1).max(),np.abs(Kv(0*s,s*LY)-1).max(),np.abs(Kv(1+0*s,s*LY)-1).max()))
