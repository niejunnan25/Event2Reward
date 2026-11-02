"""A tiny synthetic index dataset (a red block moving right) for pipeline checks.

Its labels are made up; it tests that data, model and training connect.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def make_example(destination):
    import av
    from event2reward.inference import canonical_image,history_indices
    root=Path(destination)
    if root.exists():raise FileExistsError(root)
    root.mkdir(parents=True)
    for name in ('trajectory.mp4','goal.mp4'):
        with av.open(str(root/name),'w') as container:
            stream=container.add_stream('mpeg4',rate=10);stream.width=224;stream.height=224;stream.pix_fmt='yuv420p'
            for i in range(64):
                image=Image.new('RGB',(224,224),(30,30,35));draw=ImageDraw.Draw(image)
                x=10+int(160*i/63);draw.rectangle((x,95,x+25,120),fill=(220,45,35))
                if name=='goal.mp4':draw.line((0,150,223,150),fill=(90,90,95),width=2)
                for packet in stream.encode(av.VideoFrame.from_image(image)):container.mux(packet)
            for packet in stream.encode():container.mux(packet)
    with av.open(str(root/'goal.mp4')) as container:
        goal=list(container.decode(video=0))[-1].to_ndarray(format='rgb24')
    goal=canonical_image(goal);h=hashlib.sha256()
    h.update(f'{goal.mode}:{goal.width}x{goal.height}:'.encode());h.update(np.asarray(goal).tobytes())
    def clip(frame,label,valid=False):
        return {'sample_id':f'{label}:{frame}','dataset_id':'example_media','physical_episode_id':'synthetic',
                'segment_id':'synthetic','instruction':'move the red block to the right',
                'video_path':'trajectory.mp4','history_frames':history_indices(frame),
                'start_frame':0,'goal_frame':63,'current_frame':frame,'target_progress':frame/63,
                'progress_valid':valid,'target_trend':0,
                'trend_valid':valid and len(set(history_indices(frame)))==5,'history_transform':'natural',
                'goal_available':True,'episode_end_frame':63,'goal_dataset_id':'example_media','goal_video_path':'goal.mp4',
                'metadata':{'spatial_transform':'center_square_v1','goal_contract':'language_matched_success_terminal_v1',
                            'goal_sha256':h.hexdigest(),'synthetic_pipeline_fixture':True}}
    def write(name,rows):
        (root/name).write_text(''.join(json.dumps(r)+'\n' for r in rows))
    write('expert.jsonl',[{'clips':[clip(i,'expert',True)],'relation':None} for i in [0,9,18,27,36,45,54,63]])
    indices={'expert':'expert.jsonl','expert_fill':'expert.jsonl'}
    for kind in range(3,7):
        anchors=[20,55] if kind==6 else [20];frames=anchors+[30,40]
        row={'clips':[clip(i,f'kind{kind}') for i in frames],
             'relation':{'kind':kind,'anchor_index':0,'secondary_anchor_index':1 if kind==6 else -1,
                         'member_indices':list(range(len(anchors),len(frames))),
                         'margin':0.0 if kind==3 else 0.02 if kind==5 else 0.1}}
        write(f'kind{kind}.jsonl',[row]);indices[f'event_kind{kind}']=f'kind{kind}.jsonl'
    manifest={'schema':'event2reward.dataset.v1','media_roots':{'example_media':'.'},'indices':indices}
    (root/'dataset.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return root/'dataset.json'

