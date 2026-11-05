#!/usr/bin/env python3
"""Fixed episode 0 replay from saved autonomous network frames; no simulation."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_joint_slit_followup import ARMS
from tools.run_joint_slit_learning import sha, write_json


def render(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    from matplotlib.patches import Rectangle, Circle
    from PIL import Image
    result_path = out/'final_results.json'
    if not result_path.exists():
        decision = json.loads((out/'decision.json').read_text())
        if decision['category'] != 'common_flight_not_established':
            raise RuntimeError('No completed experiment outcome for video generation')
        write_json(out/'demo_status.json', dict(produced=False, reason='Common flight gate failed; joint arms not started.'))
        return
    result = json.loads(result_path.read_text())
    successes = result.get('successes', {})
    camera_task_gain = successes.get(ARMS[1],0)-successes.get(ARMS[0],0) >= 4
    if not result['joint_candidate'] and not camera_task_gain:
        write_json(out/'demo_status.json', dict(produced=False,
            reason='No interpretable task-gain/coordination candidate; do not substitute a scripted demonstration.'))
        return
    data = []
    for arm in ARMS[:2]:
        directory = out/'arms'/arm/'confirmation'
        row = json.loads((directory/'results.json').read_text())['rows'][0]
        trace = json.loads((directory/'results_trace_000.json').read_text())
        with np.load(directory/'results_frames_000.npz') as archive:
            frames = archive['frames'].copy()
        if len(frames) != len(trace) or not trace:
            raise ValueError('Missing or mismatched real frame/trajectory recording')
        data.append(dict(arm=arm,row=row,trace=trace,frames=frames))
    if data[0]['row']['episode_seed'] != data[1]['row']['episode_seed'] or data[0]['row']['slit_center_y'] != data[1]['row']['slit_center_y']:
        raise ValueError('Video policies are not paired on the same layout')
    demo = out/'demo'
    demo.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':13,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.labelcolor':'#344254','text.color':'#233348'})
    fig = plt.figure(figsize=(19.2,10.8),dpi=100,facecolor='white')
    grid = fig.add_gridspec(2,3,width_ratios=(1.25,1,1),height_ratios=(1.35,1),
                           left=.055,right=.97,bottom=.09,top=.88,wspace=.27,hspace=.30)
    flight = fig.add_subplot(grid[:,0])
    views = [fig.add_subplot(grid[0,i]) for i in (1,2)]
    exposure_ax = fig.add_subplot(grid[1,1])
    gain_ax = fig.add_subplot(grid[1,2])
    colors = ('#777f89','#28658a')
    names = ('Automatic exposure','Learned camera + flight')
    center = data[0]['row']['slit_center_y']
    for low,high in ((-1.2,center-.15),(center+.15,1.2)):
        flight.add_patch(Rectangle((-.05,low),.1,high-low,fc='#c1c7ce',ec='none'))
    for side in (-1,1):
        flight.axhline(side*1.2,color='#d5d9df',lw=3)
    flight.scatter([1.5],[0.],marker='*',s=220,color='#56826b',zorder=4,label='Goal')
    flight.set(xlim=(-2.8,1.9),ylim=(-1.45,1.45),xlabel='Forward position (m)',ylabel='Lateral position (m)',title='Top view')
    flight.set_aspect('equal',adjustable='box')
    images, lines, bodies, statuses = [],[],[],[]
    for d,ax,color,name in zip(data,views,colors,names):
        images.append(ax.imshow(d['frames'][0],cmap='gray',vmin=0,vmax=1,interpolation='nearest'))
        ax.set_title(name,pad=14)
        ax.set_axis_off()
        statuses.append(ax.text(.02,-.10,'',transform=ax.transAxes,color=color,fontsize=12))
        line, = flight.plot([],[],color=color,lw=2.5,label=name)
        lines.append(line)
        body = Circle((-2.5,0),.12,fc=color,ec='white',lw=1.2,zorder=5,alpha=.9)
        flight.add_patch(body)
        bodies.append(body)
        times = [r['time_s'] for r in d['trace']]
        exposure_ax.plot(times,[r['exposure_us']/1000 for r in d['trace']],color=color,lw=2)
        gain_ax.plot(times,[r['gain'] for r in d['trace']],color=color,lw=2)
    flight.legend(loc='lower left',fontsize=10,frameon=False)
    exposure_ax.set(title='Exposure',xlabel='Flight time (s)',ylabel='Milliseconds',xlim=(0,6))
    gain_ax.set(title='Gain',xlabel='Flight time (s)',ylabel='Normalized setting',xlim=(0,6),ylim=(0,1))
    cursors = [ax.axvline(0,color='#c59252',lw=1.5) for ax in (exposure_ax,gain_ax)]
    for ax in (flight,exposure_ax,gain_ax):
        ax.grid(alpha=.15)
    fig.text(.055,.955,'Task-driven camera and flight',fontsize=25,weight='medium')
    fig.text(.055,.915,'Autonomous neural policies · paired episode 0 · simulation · 0.3× playback',fontsize=13,color='#65717e')
    def draw(time):
        for d,image,line,body,status in zip(data,images,lines,bodies,statuses):
            times = np.array([r['time_s'] for r in d['trace']])
            index = int(np.clip(np.searchsorted(times,time,side='right')-1,0,len(times)-1))
            image.set_data(d['frames'][index])
            trace = d['trace'][:index+1]
            xs,ys = [r['x'] for r in trace],[r['y'] for r in trace]
            if time >= d['row']['elapsed_s']:
                post = d['trace'][-1]['post_position']
                xs.append(post[0]);ys.append(post[1])
                status.set_text({'goal':'Arrived','collision':'Collision','timeout':'Time limit'}[d['row']['stop_reason']])
            else:
                status.set_text('')
            line.set_data(xs,ys)
            body.center=(xs[-1],ys[-1])
        for cursor in cursors:
            cursor.set_xdata([time,time])
    draw(0)
    preview = demo/'preview_full.png'
    fig.savefig(preview,dpi=100)
    with Image.open(preview) as image:
        image.resize((960,540)).convert('RGB').save(demo/'preview.jpg',quality=83)
    preview.unlink()
    video = demo/'joint_slit_learned_episode0.mp4'
    writer = FFMpegWriter(fps=30,codec='libx264',extra_args=['-crf','18','-pix_fmt','yuv420p'])
    with writer.saving(fig,str(video),100):
        for time in np.linspace(0,6,600):
            draw(float(time))
            writer.grab_frame(facecolor='white')
    plt.close(fig)
    write_json(out/'demo_status.json',dict(produced=True,path=str(video),sha256=sha(video),
        episode_seed=data[0]['row']['episode_seed'],fixed_episode_index=0,width=1920,height=1080,
        duration_seconds=20,scripted_actions=False,source_sha256=sha(__file__),joint_candidate=result['joint_candidate'],
        caveat='Fixed paired replay, not a representative-success claim; raw sensor resolution is 96x72.'))
    print(video,flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out_dir',type=Path,required=True)
    render(parser.parse_args().out_dir.resolve())
