#!/usr/bin/env python3
"""Render a 20-second, explicitly diagnostic demonstration from saved frames."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FFMpegWriter


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',required=True);c=p.parse_args()
    replay_path=Path(c.data).parent/("demo_ae_replay.json" if Path(c.data).stem=="demo_ae" else "confirmation_0.json")
    terminal=json.loads(replay_path.read_text())["time_s"][0]
    terminal_index=min(119,int(round(terminal*15)))
    data=np.load(c.data);trajectory=data['trajectory'];frames=data['frames'];settings=data['settings']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':14,'axes.spines.top':False,'axes.spines.right':False})
    fig=plt.figure(figsize=(16,9),dpi=120,facecolor='#ffffff')
    fig.suptitle('Move to reveal. Observe to choose.',x=.08,y=.95,ha='left',fontsize=27,color='#152c43')
    fig.text(.08,.90,'SCRIPTED DIAGNOSTIC + MEAN AUTO-EXPOSURE  |  NOT A LEARNED POLICY',fontsize=12,color='#6b7785')
    ax=fig.add_axes([.07,.17,.42,.65]);im=fig.add_axes([.56,.25,.39,.52])
    for x,y,w,h in [(0,-.55,2.2,1.1),(1.7,-1.4,.2,.85),(-2,-1.6,6,.2),(-2,1.4,6,.2)]:
        ax.add_patch(Rectangle((x,y),w,h,color='#657482' if x==0 else '#bac2c9'))
    ax.set(xlim=(-1.4,3.4),ylim=(-1.7,1.7),xlabel='Forward (m)',ylabel='Lateral (m)');ax.set_aspect('equal')
    ax.scatter([3],[0],marker='*',s=180,color='#b88838',label='Goal')
    path,=ax.plot([],[],lw=2.8,color='#256c92');dot,=ax.plot([],[],'o',ms=11,color='#256c92')
    ax.axvline(1.8,color='#b88838',alpha=.3,ls=':')
    shown=im.imshow(frames[0],cmap='gray',vmin=0,vmax=1,interpolation='nearest');im.axis('off')
    im.set_title('Observed grayscale frame',loc='left',pad=20,color='#152c43')
    text=fig.text(.56,.18,'',fontsize=17,color='#152c43');phase=fig.text(.07,.09,'',fontsize=17,color='#256c92')
    output=Path(c.out);output.parent.mkdir(parents=True,exist_ok=True)
    writer=FFMpegWriter(fps=6,codec='libx264',extra_args=['-crf','17','-pix_fmt','yuv420p'])
    with writer.saving(fig,str(output),120):
        for k in range(len(frames)):
            # Episode 0 reaches its goal before the 8-second deadline; hold terminal view.
            j=min(k,terminal_index)
            path.set_data(trajectory[:j+1,0],trajectory[:j+1,1]);dot.set_data([trajectory[j,0]],[trajectory[j,1]])
            shown.set_data(frames[j]);text.set_text(f'Exposure command {settings[j,0]:.2f}   |   Gain command {settings[j,1]:.2f}')
            phase.set_text('1  Lateral observation' if j<24 else '2  Image-based route choice  →  scripted navigation')
            writer.grab_frame()
    plt.close(fig)
if __name__=='__main__':main()
