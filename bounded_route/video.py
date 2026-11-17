"""Export the fixed development-id0 example; always label it as simulation."""
from pathlib import Path
import subprocess
import torch
from PIL import Image,ImageDraw


def export_videos(out):
    out=Path(out)
    for joint in ('C','D'):
        data=torch.load(out/'examples'/f'{joint}_joint.pth',map_location='cpu',weights_only=False)
        frames=[]
        for row in data['timeline']:
            gray=(row['frame'][0,0].clamp(0,1).numpy()*255).astype('uint8')
            photo=Image.fromarray(gray).resize((576,432),Image.Resampling.NEAREST).convert('RGB')
            canvas=Image.new('RGB',(640,520),'white');canvas.paste(photo,(32,40));draw=ImageDraw.Draw(canvas)
            draw.text((15,10),f'Research simulation | {joint} | fixed development episode 0',fill='black')
            draw.text((15,480),f"t={row['time_s']:.2f}s exposure={row['exposure_us'][0]/1000:.2f}ms gain={row['gain'][0]:.2f}",fill='black')
            frames.append(canvas.tobytes())
        proc=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','640x520',
             '-r','15','-i','-','-an','-c:v','libx264','-pix_fmt','yuv420p',str(out/f'{joint}_example.mp4')],stdin=subprocess.PIPE)
        try:
            for frame in frames:proc.stdin.write(frame)
        finally:proc.stdin.close()
        if proc.wait()!=0:raise RuntimeError(f'Video export failed: {joint}')
