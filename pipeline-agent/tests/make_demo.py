"""Create CPU-only demo resources; the small sample override is simulation-only."""
import sys,json
from pathlib import Path
from PIL import Image
p=Path(sys.argv[1]).resolve();p.mkdir(parents=True,exist_ok=True)
v=p/'tiny.gif'
Image.new('RGB',(32,32),'red').save(v,save_all=True,append_images=[Image.new('RGB',(32,32),'blue')],duration=100)
for name,color in [('f0','red'),('final','blue')]:
    Image.new('RGB',(512,512),color).save(p/f'{name}.png')
c=dict(mode='simulation',task='shape_color_then_scale',simulation_samples=1,
       sources={'tiny':dict(video=str(v),f0=str(p/'f0.png'),final=str(p/'final.png'),frame_count=2,
                            description='Synthetic demo',eval_text='<image>\nSynthetic demo')})
(p/'demo.json').write_text(json.dumps(c),encoding='utf-8')
