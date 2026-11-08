"""Independent, read-only numerical review of X1-a. No production fixes or training.

Run from the repository root with --output-dir pointing to a new directory.
The cell-by-cell oracle deliberately does not use solver_wall.fluid_cells or
its vectorised conductance/volume expressions. See joint record LOG-075.

The saved results in results/x1/codex_review are the LOG-075 snapshot, when point
contacts and unanchored regions were still accepted. Since LOG-077/081 the solver
rejects them before solving; those cases below now record the rejection (LOG-105).
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import resource
import sys
import time
import warnings
from unittest import mock

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from classroom_heat_pinn_data.synthetic import solver, solver_wall as sw

WALL = {"x0": 0., "x1": .6, "y0": .3, "y1": .34}
BI = {"S": 1., "E": .4, "N": .25}
TINF = {"S": 0., "E": 0., "N": .7}


def independent_cells(nx, solids, bi):
    x, y, h = solver.make_grid(nx)
    ny = len(y)
    volume = np.zeros((nx, ny))
    boundary = {w: np.zeros((nx, ny)) for w in sw.WALLS}
    edges = {}
    for i in range(nx - 1):
        for j in range(ny - 1):
            midx, midy = (x[i] + x[i+1])/2, (y[j] + y[j+1])/2
            if any(s['x0'] < midx < s['x1'] and s['y0'] < midy < s['y1'] for s in solids):
                continue
            a, b, c, d = (i, j), (i+1, j), (i, j+1), (i+1, j+1)
            for node in (a, b, c, d):
                volume[node] += h*h/4
            for u, v in ((a,b), (c,d), (a,c), (b,d)):
                key = (u, v)
                edges[key] = edges.get(key, 0.) + .5
            for condition, wall, nodes in ((j == 0,'S',(a,b)), (j == ny-2,'N',(c,d)),
                                           (i == 0,'W',(a,c)), (i == nx-2,'E',(b,d))):
                if condition:
                    for node in nodes:
                        boundary[wall][node] += h/2
    active = volume > 0
    index = np.full(active.shape, -1)
    index[active] = np.arange(active.sum())
    diag = np.zeros(active.sum())
    rows, cols, values = [], [], []
    for (u,v), conductance in edges.items():
        a, b = index[u], index[v]
        rows.extend((a,b)); cols.extend((b,a)); values.extend((-conductance,-conductance))
        diag[a] += conductance; diag[b] += conductance
    for wall, length in boundary.items():
        diag += bi.get(wall,0) * length[active]
    rows.extend(range(len(diag))); cols.extend(range(len(diag))); values.extend(diag)
    matrix = sp.csr_matrix((values,(rows,cols)),shape=(len(diag),len(diag)))
    return volume, boundary, active, matrix


def structural_checks():
    geometries = [[], [WALL], [{"x0":.2,"x1":.4,"y0":.2,"y1":.4}],
                  [{"x0":.2,"x1":.4,"y0":0.,"y1":.2}],
                  [{"x0":.8,"x1":1.,"y0":.2,"y1":.4}],
                  [{"x0":.2,"x1":.4,"y0":.6,"y1":.8}],
                  [{"x0":0.,"x1":.4,"y0":.2,"y1":.4},
                   {"x0":.2,"x1":.6,"y0":.2,"y1":.4}]]
    bi = {"S":.8,"N":.6,"W":.7,"E":.4}
    ambient = {"S":.2,"N":.7,"W":.3,"E":.1}
    forcing = {w: (lambda s: .01 + .02*s) for w in bi}
    records = []
    for solids in geometries:
        x,y,h = solver.make_grid(51)
        X,Y = np.meshgrid(x,y,indexing='ij')
        q = 1 + X + 2*Y
        _,_,theta,info = sw.solve(51,bi,ambient,solids=solids,Q=q,g=forcing)
        vol,length,active,matrix = independent_cells(51,solids,bi)
        np.testing.assert_array_equal(active,info['active'])
        np.testing.assert_allclose(vol,info['vol'],atol=1e-18,rtol=1e-14)
        difference = matrix-info['matrix']
        assert difference.nnz == 0 or abs(difference.data).max() < 1e-14
        rhs = q*vol
        for wall, coord, sl in [('S',x,np.s_[:,0]),('N',x,np.s_[:,-1]),
                                ('W',y,np.s_[0,:]),('E',y,np.s_[-1,:])]:
            rhs[sl] += (bi[wall]*ambient[wall]-forcing[wall](coord))*length[wall][sl]
            flux = np.sum((bi[wall]*(theta[sl][length[wall][sl]>0]-ambient[wall]) +
                           forcing[wall](coord[length[wall][sl]>0]))*length[wall][sl][length[wall][sl]>0])
            np.testing.assert_allclose(flux,info['total_flux'][wall],atol=1e-13)
        residual = matrix @ theta[active]-rhs[active]
        assert abs(residual).max() < 1e-12
        assert abs(float(vol.sum()) - sum(.0004 for i in range(50) for j in range(40)
                   if not any(s['x0']<(i+.5)*h<s['x1'] and s['y0']<(j+.5)*h<s['y1'] for s in solids))) < 1e-12
        if solids == [WALL]:
            np.testing.assert_allclose(vol.sum(), .8-.6*.04,atol=1e-14)
            expected={'S':1.,'N':1.,'E':.8,'W':.76}
            for w in length:
                np.testing.assert_allclose(length[w].sum(),expected[w],atol=1e-14)
        records.append({'solids':solids,'fluid_area':float(vol.sum()),
                        'outer_lengths':{w:float(v.sum()) for w,v in length.items()},
                        'matrix_maxdiff':float(abs(difference.data).max()) if difference.nnz else 0.,
                        'independent_residual_max':float(abs(residual).max())})
    return records


def mms_checks():
    hs,errors,orders=sw.mms_wall_test(WALL)
    assert min(orders[-2:]) >= 1.8
    p=np.pi/.6
    flux_normals={'lower':float(np.max(abs(20*((.3-.32)**2-.02**2)))),
                  'upper':float(np.max(abs(20*((.34-.32)**2-.02**2)))),
                  'tip':float(abs(-.3*p*np.sin(p*.6)))}
    assert max(flux_normals.values()) < 1e-13
    # An intentionally broken solver that ignores the obstacle entirely.
    def no_wall(nx,bi,tinf,solids=None,**kwargs):
        x,y,t,info=solver.solve(nx,bi,tinf,**kwargs)
        info['active']=np.ones_like(t,dtype=bool)
        return x,y,t,info
    with mock.patch.object(sw,'solve',no_wall):
        _,broken_errors,broken_orders=sw.mms_wall_test(WALL)
    # A disconnected, two-temperature exact solution checks wall presence directly.
    solid={'x0':.4,'x1':.6,'y0':0.,'y1':.8}
    bi={w:1. for w in sw.WALLS};ambient={w:0. for w in sw.WALLS}
    g={'S':lambda x: -np.where(x<=.4,1.,2.),'N':lambda x: -np.where(x<=.4,1.,2.),
       'W':lambda y: -np.ones_like(y),'E':lambda y: -2*np.ones_like(y)}
    x,y,t,info=sw.solve(51,bi,ambient,solids=[solid],g=g)
    exact=np.broadcast_to(np.where(x[:,None]<=.4,1.,2.),t.shape)
    err=float(abs(t[info['active']]-exact[info['active']]).max());assert err<1e-12
    _,_,bad,_=no_wall(51,bi,ambient,solids=[solid],g=g)
    broken_err=float(abs(bad[info['active']]-exact[info['active']]).max())
    assert broken_err>.1
    return {'wall_normal_derivatives':flux_normals,'errors':errors,'orders':orders,
            'wall_ignored_errors':broken_errors,'wall_ignored_orders':broken_orders,
            'wall_ignored_passes_original_MMS':min(broken_orders[-2:])>=1.8,
            'two_component_exact_error':err,'two_component_wall_ignored_error':broken_err}


def edge_cases():
    # Two fluid regions that only touch at a single vertex are distinct open regions.
    solids=[{'x0':0.,'x1':.5,'y0':.4,'y1':.8},{'x0':.5,'x1':1.,'y0':0.,'y1':.4}]
    try:
        x,y,t,info=sw.solve(51,BI,TINF,solids=solids,heaters=[])
    except ValueError as exc:  # current contract (LOG-077/081): rejected before solving
        result={'point_touch_geometry':{'accepted':False,'rejected':str(exc)},'invalid_rectangles':[]}
    else:
        fluid=info['fluid_cells'].astype(bool)
        from scipy.ndimage import label
        n_fluid=int(label(fluid,structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))[1])
        n_graph=int(connected_components(info['matrix'],directed=False,return_labels=False))
        result={'point_touch_geometry':{'fluid_components':n_fluid,'solver_components':n_graph,
                                        'accepted':True},'invalid_rectangles':[]}
    for solid in [{'x0':.6,'x1':.2,'y0':.3,'y1':.34},
                  {'x0':0.,'x1':.6,'y0':.3,'y1':.3}]:
        try:
            _,_,_,i=sw.solve(51,BI,TINF,solids=[solid])
            result['invalid_rectangles'].append({'solid':solid,'accepted':True,'fluid_area':float(i['vol'].sum())})
        except ValueError as exc:
            result['invalid_rectangles'].append({'solid':solid,'accepted':False,'error':str(exc)})
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            _,_,u,i=sw.solve(51,{'E':1.},{},solids=[{'x0':.4,'x1':.6,'y0':0.,'y1':.8}],
                           heaters=[{'x':.2,'y':.4,'s':.05,'A':1.}])
    except ValueError as exc:  # current contract (LOG-077): rejected before solving
        result['unanchored_component']={'accepted':False,'rejected':str(exc)}
    else:
        result['unanchored_component']={'warnings':[str(w.message) for w in caught],
                                       'finite_output':bool(np.isfinite(u[i['active']]).all()),
                                       'maxabs':float(abs(u[np.isfinite(u)]).max()) if np.isfinite(u).any() else None,
                                       'balance_rel_err':i['balance_rel_err'] if np.isfinite(i['balance_rel_err']) else None}
    x,y,u,i=sw.solve(101,BI,TINF,solids=[WALL],heaters=[])
    pts=np.array([[.3,.3],[.3,.34],[.6,.32],[.3,.4]])
    sampled=solver.sample(x,y,u,pts)
    result['existing_rectangular_sampler']={'points':pts.tolist(),'finite':np.isfinite(sampled).tolist()}
    return result


def refinement(max_nx):
    known=json.loads((ROOT/'classroom_heat_pinn_data/synthetic/data/known.json').read_text())
    import pandas as pd
    df=pd.read_csv(ROOT/'classroom_heat_pinn_data/synthetic/data/sensors_A_clean.csv')
    pts=df[['x','y']].to_numpy()
    previous=None;records=[];sensors={};diffs=[]
    for nx in (101,201,401,801,1601):
        if nx>max_nx:break
        start=time.monotonic()
        x,y,u,info=sw.solve(nx,BI,TINF,solids=[WALL],heaters=[known['heater']])
        active=info['active'];h=info['h']
        nodes=np.rint(pts/h).astype(int)
        values=u[nodes[:,0],nodes[:,1]];assert np.isfinite(values).all()
        sensors[nx]=values.tolist()
        rec={'nx':nx,'active_nodes':int(active.sum()),'seconds':time.monotonic()-start,
             'energy_relative_error':info['balance_rel_err'],'heater_power':info['q_in']}
        if previous is not None:
            oldnx,ox,oy,old=previous
            d=abs(old-u[::2,::2]);X,Y=np.meshgrid(ox,oy,indexing='ij')
            distance=np.minimum(np.hypot(X-.6,Y-.3),np.hypot(X-.6,Y-.34))
            k=np.nanargmax(d);ix,iy=np.unravel_index(k,d.shape)
            diffs.append({'coarse':oldnx,'fine':nx,'max_theta':float(np.nanmax(d)),
                          'rms_theta':float(np.sqrt(np.nanmean(d*d))),
                          'max_away_0p1_theta':float(np.nanmax(d[distance>.1])),
                          'max_location':[float(ox[ix]),float(oy[iy])],
                          'sensor_max_theta':float(abs(np.array(sensors[oldnx])-values).max()),
                          'sensor_rms_theta':float(np.sqrt(np.mean((np.array(sensors[oldnx])-values)**2)))})
        previous=(nx,x,y,u)
        del info
        gc.collect()
        records.append(rec);print('GRID',json.dumps(rec),flush=True)
    ref=np.array(sensors[max(sensors)])
    sensor_diffs={str(nx):{'max_C':float(20*abs(np.array(v)-ref).max()),
                          'rms_C':float(20*np.sqrt(np.mean((np.array(v)-ref)**2)))} for nx,v in sensors.items()}
    observed=[]
    for a,b in zip(diffs,diffs[1:]):
        order=float(np.log2(a['max_theta']/b['max_theta']))
        observed.append({'fine':b['fine'],'max_difference_order':order,
                         'fine_error_Richardson_theta_estimate':b['max_theta']/(2**order-1)})
    return {'grids':records,'differences':diffs,'observed':observed,
            'sensor_vs_finest':sensor_diffs,'sensor_values_theta':sensors,
            'peak_rss_bytes_macos':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def supplementary_checks():
    from scipy.special import erf
    known=json.loads((ROOT/'classroom_heat_pinn_data/synthetic/data/known.json').read_text())
    ht=known['heater']
    def integral(x0,x1,y0,y1):
        def part(lo,hi,center):
            return ht['s']*np.sqrt(np.pi/2)*(erf((hi-center)/(np.sqrt(2)*ht['s']))-
                                             erf((lo-center)/(np.sqrt(2)*ht['s'])))
        return ht['A']*part(x0,x1,ht['x'])*part(y0,y1,ht['y'])
    whole=integral(0,1,0,.8);removed=integral(0,.6,.3,.34)
    # Nonseparable smooth MMS also exercises nonzero W Robin plus W forcing.
    p=np.pi/.6;a=.3;b=20.;c=5.;d=.02
    def exact(X,Y):
        z=Y-.32
        return 1+a*np.cos(p*X)*(1+c*(z*z-d*d)**2)+b*(z**3/3-d*d*z)
    def dx(X,Y):return -a*p*np.sin(p*X)*(1+c*((Y-.32)**2-d*d)**2)
    def dy(X,Y):
        z=Y-.32
        return a*np.cos(p*X)*4*c*z*(z*z-d*d)+b*(z*z-d*d)
    def source(X,Y):
        z=Y-.32
        return a*p*p*np.cos(p*X)*(1+c*(z*z-d*d)**2)-a*np.cos(p*X)*c*(12*z*z-4*d*d)-2*b*z
    bi={'S':5.,'E':1.,'N':.5,'W':.8};ambient={'S':0.,'E':0.,'N':.9,'W':.2}
    g={'S':lambda x:dy(x,0*x)-bi['S']*(exact(x,0*x)-ambient['S']),
       'N':lambda x:-dy(x,.8+0*x)-bi['N']*(exact(x,.8+0*x)-ambient['N']),
       'E':lambda y:-dx(1+0*y,y)-bi['E']*(exact(1+0*y,y)-ambient['E']),
       'W':lambda y:dx(0*y,y)-bi['W']*(exact(0*y,y)-ambient['W'])}
    errors=[]
    for nx in (51,101,201,401):
        x,y,_=solver.make_grid(nx);X,Y=np.meshgrid(x,y,indexing='ij')
        _,_,u,i=sw.solve(nx,bi,ambient,solids=[WALL],Q=source(X,Y),g=g)
        errors.append(float(abs(u[i['active']]-exact(X,Y)[i['active']]).max()))
    orders=np.log2(np.array(errors[:-1])/errors[1:]).tolist();assert min(orders[-2:])>=1.8
    pinch=[{'x0':0.,'x1':.5,'y0':.4,'y1':.8},{'x0':.5,'x1':1.,'y0':0.,'y1':.4}]
    out={'exact_heater_power_no_wall':whole,'exact_heater_power_with_wall':whole-removed,
         'removed_heater_fraction':removed/whole,'nonseparable_MMS_errors':errors,'nonseparable_MMS_orders':orders}
    try:
        _,_,u,i=sw.solve(51,BI,TINF,solids=pinch)
    except ValueError as exc:  # current contract (LOG-077/081)
        out['point_contact_rejected']=str(exc)
        return out
    # Lower-left component has no source or warm external boundary; true solution is zero there.
    out.update(point_contact_cold_component_S_flux=i['total_flux']['S'],
               point_contact_cold_component_temperature_max=float(np.nanmax(u[:25,:20])))
    return out


def sensor_diagnostic(out):
    import pandas as pd
    known=json.loads((ROOT/'classroom_heat_pinn_data/synthetic/data/known.json').read_text())
    df=pd.read_csv(ROOT/'classroom_heat_pinn_data/synthetic/data/sensors_A_clean.csv')
    pts=df[['x','y']].to_numpy()
    grids=json.loads((out/'refinement.json').read_text())
    finest=max(map(int,grids['sensor_values_theta']))
    reference=np.array(grids['sensor_values_theta'][str(finest)])
    result={'purpose':'truth-input local tool diagnostic, not inverse fitting or global identifiability',
            'reference_nx':finest,'grids':[]}
    for nx in (101,201,401):
        def predict(beta):
            bi=dict(zip(('S','E','N'),np.exp(beta)))
            x,y,u,info=sw.solve(nx,bi,TINF,solids=[WALL],heaters=[known['heater']])
            ij=np.rint(pts/info['h']).astype(int)
            return u[ij[:,0],ij[:,1]]
        beta=np.log([1.,.4,.25]);base=predict(beta);jacobians=[]
        for step in (.001,.0001):
            J=np.column_stack([(predict(beta+step*e)-predict(beta-step*e))/(2*step) for e in np.eye(3)])
            jacobians.append(J)
        J=jacobians[-1];u,s,vh=np.linalg.svd(J,full_matrices=False)
        rank=int(sum(s>s[0]*1e-10));assert rank==3
        delta=vh.T@((u.T@(reference-base))/s)
        cov=(vh.T*(.005/s)**2)@vh
        result['grids'].append({'nx':nx,'rank':rank,'condition':float(s[0]/s[-1]),
                               'singular_values_theta_per_logBi':s.tolist(),
                               'relative_std_at_sigma_0p1C':np.sqrt(np.diag(cov)).tolist(),
                               'linearized_grid_bias_relative_Bi':np.expm1(delta).tolist(),
                               'jacobian_step_difference_max':float(abs(jacobians[0]-J).max())})
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--mode',choices=['structure','refinement','supplement','sensors'],default='structure')
    parser.add_argument('--max-nx',type=int,choices=[801,1601],default=801)
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    target=args.output_dir/(args.mode+'.json')
    if target.exists():raise FileExistsError(target)
    if args.mode=='structure':
        result={'independent_cell_assembly':structural_checks(),'MMS':mms_checks(),'edge_cases':edge_cases()}
    elif args.mode=='refinement':result=refinement(args.max_nx)
    elif args.mode=='supplement':result=supplementary_checks()
    else:result=sensor_diagnostic(args.output_dir)
    result['source_sha256']=hashlib.sha256((ROOT/'classroom_heat_pinn_data/synthetic/solver_wall.py').read_bytes()).hexdigest()
    target.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print('RESULT',json.dumps(result,allow_nan=False),flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
