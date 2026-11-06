"""Render the supplied truth fields; importing this module performs no I/O."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA = BASE_DIR / "data"
DEFAULT_OUTPUT = BASE_DIR / "generated" / "data_overview.png"


def plot_overview(data_dir=DEFAULT_DATA, output=DEFAULT_OUTPUT):
    data_dir, output = Path(data_dir).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Existing figure: {output}; choose a new --output")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with ExitStack() as stack:
        cfg=json.loads((data_dir / "config.json").read_text())
        R=stack.enter_context(np.load(data_dir / "reference.npz"))
        W=stack.enter_context(np.load(data_dir / "whatif_reference.npz"))
        M=stack.enter_context(np.load(data_dir / "mismatch_reference.npz"))
        A=np.array(cfg['sensors']['layout_A']); B=np.array(cfg['sensors']['layout_B'])
        ext=[0,1*10,0,0.8*10]
        fig,ax=plt.subplots(1,3,figsize=(15,4.3),constrained_layout=True)
        vmin,vmax=10,28
        def room(a,T,title):
            im=a.imshow(T.T,origin='lower',extent=ext,cmap='RdYlBu_r',vmin=vmin,vmax=vmax,aspect='equal')
            a.set_title(title); a.set_xlabel('x [m]'); a.set_ylabel('y [m]')
            kw=dict(fontsize=7,bbox=dict(boxstyle='round,pad=0.15',fc='white',ec='none',alpha=0.8))
            a.text(5,0.15,'S: window wall',ha='center',va='bottom',**kw)
            a.text(5,7.85,'N: corridor wall',ha='center',va='top',**kw)
            a.text(9.85,4,'E: exterior',rotation=90,ha='right',va='center',**kw)
            a.text(0.15,4,'W: adiabatic',rotation=90,ha='left',va='center',**kw)
            return im
        im=room(ax[0],R['T_C'],'Reference (truth)')
        ax[0].scatter(A[:,0]*10,A[:,1]*10,s=40,c='k',marker='o',label='layout A')
        ax[0].scatter(B[:,0]*10,B[:,1]*10,s=40,c='k',marker='x',label='layout B')
        h=cfg['heater']; ax[0].scatter([h['x']*10],[h['y']*10],s=120,marker='*',c='w',edgecolors='k',label='heater')
        ax[0].legend(loc='upper right',fontsize=7)
        room(ax[1],W['T_C'],'What-if: heater moved to window')
        hw=cfg['heater_whatif']; ax[1].scatter([hw['x']*10],[hw['y']*10],s=120,marker='*',c='w',edgecolors='k')
        fig.colorbar(im,ax=ax[:2],label='T [°C]',shrink=0.85)
        d=M['T_C']-R['T_C']; lim=np.abs(d).max()
        im2=ax[2].imshow(d.T,origin='lower',extent=ext,cmap='Blues_r',vmin=-lim,vmax=0,aspect='equal')
        ax[2].set_title('Mismatch "reality" minus simple model'); ax[2].set_xlabel('x [m]')
        dr=cfg['mismatch']['draft']; ax[2].scatter([dr['x']*10],[dr['y']*10],s=80,marker='v',c='k',label='window draft')
        ax[2].legend(loc='upper right',fontsize=7); ax[2].set_ylabel('y [m]')
        fig.colorbar(im2,ax=ax[2],label='ΔT [°C]',shrink=0.85)
        output.parent.mkdir(parents=True, exist_ok=True)
        try:
            fig.savefig(output, dpi=140)
        finally:
            plt.close(fig)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    print(plot_overview(args.data_dir, args.output))


if __name__ == "__main__":
    main()
