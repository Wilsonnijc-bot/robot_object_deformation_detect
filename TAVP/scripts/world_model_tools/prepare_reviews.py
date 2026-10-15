#!/usr/bin/env python3
"""Prepare dense, timestamped visual review sheets (requires Pillow).

Temporary sheets cover every second decoded frame of each complete source video.
Dobot review sources contain only the center third, at original panel resolution.
"""
import argparse
import json
import math
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]


def prepare(folder, output):
    metadata = json.loads((folder / 'original/metadata/metadata.json').read_text())
    work = output / metadata['video_id']
    work.mkdir(parents=True, exist_ok=True)
    source = folder / 'original/video.mp4'
    crop = None
    if metadata['robot_type'].startswith('Dobot'):
        width = metadata['resolution']['width'] // 3
        height = metadata['resolution']['height']
        crop = {'x': width, 'y': 0, 'width': width, 'height': height}
        source = work / 'middle_view.mp4'
        if not source.exists():
            subprocess.run(['ffmpeg','-v','error','-xerror','-y','-i',str(folder/'original/video.mp4'),
                            '-map','0:v:0','-vf',f'crop={width}:{height}:{width}:0',
                            '-c:v','libx264','-crf','16','-preset','medium','-pix_fmt','yuv420p',
                            '-fps_mode','passthrough','-an','-movflags','+faststart',str(source)],check=True)
    info = json.loads(subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_streams','-of','json',str(source)]))['streams'][0]
    first = work/'initial.png'
    # Extract directly from original, before derivative encoding changes pixels.
    cmd = ['ffmpeg','-v','error','-y','-noautorotate','-i',str(folder/'original/video.mp4'),'-frames:v','1']
    if crop:
        cmd += ['-vf',f"crop={crop['width']}:{crop['height']}:{crop['x']}:{crop['y']}"]
    subprocess.run(cmd+['-pix_fmt','rgb24',str(first)],check=True)
    thumb_width = 240
    thumb_height = round(info['height']*thumb_width/info['width'])
    cell_height = thumb_height + 20
    cols, rows, stride = 8, 8, 2
    font = ImageFont.load_default(size=13)
    sheets = []
    command = ['ffmpeg','-v','error','-xerror','-noautorotate','-i',str(source),
               '-vf',f'select=not(mod(n\\,2)),scale={thumb_width}:{thumb_height}',
               '-fps_mode','passthrough','-pix_fmt','rgb24','-f','rawvideo','-']
    process = subprocess.Popen(command,stdout=subprocess.PIPE)
    frame_bytes = thumb_width*thumb_height*3
    count = 0
    sheet = None
    fps = metadata['fps']
    while True:
        data = process.stdout.read(frame_bytes)
        if not data:
            break
        if len(data)!=frame_bytes:
            raise RuntimeError('Incomplete review frame')
        slot = count % (cols*rows)
        if slot==0:
            sheet = Image.new('RGB',(cols*thumb_width,rows*cell_height+28),'#161616')
            draw = ImageDraw.Draw(sheet)
            draw.text((8,5),f"Video {metadata['video_id']} | every 2nd source frame | source FPS {fps:g} | row-major chronological order",font=font,fill='white')
        x=(slot%cols)*thumb_width
        y=(slot//cols)*cell_height+28
        sheet.paste(Image.frombytes('RGB',(thumb_width,thumb_height),data),(x,y))
        frame=count*stride
        draw.text((x+4,y+thumb_height+2),f'f{frame:04d}  {frame/fps:06.2f}s',font=font,fill='white')
        count+=1
        if count%(cols*rows)==0:
            path=work/f'sheet_{len(sheets)+1:03d}.jpg'
            sheet.save(path,quality=90)
            sheets.append(str(path))
    if count%(cols*rows):
        path=work/f'sheet_{len(sheets)+1:03d}.jpg'
        sheet.save(path,quality=90)
        sheets.append(str(path))
    if process.wait()!=0:
        raise RuntimeError('Review frame decoding failed')
    assert count == math.ceil(metadata['decoded_frame_count']/2)
    index={'video_id':metadata['video_id'],'folder':str(folder),'source_video':str(source),
           'original_video':str(folder/'original/video.mp4'),'initial_png':str(first),'crop':crop,
           'fps':fps,'source_frame_count':metadata['decoded_frame_count'],
           'inspected_frame_stride':2,'sampled_frame_count':count,'contact_sheets':sheets,
           'sheet_rows':rows,'sheet_columns':cols,'sheet_thumbnail_width':thumb_width,
           'sheet_thumbnail_height':thumb_height}
    (work/'review_index.json').write_text(json.dumps(index,indent=2)+'\n')
    print(f"READY {metadata['video_id']}: {len(sheets)} sheets, {count} frames, {source}",flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path('/tmp/tavp-dense-reviews'))
    parser.add_argument('--ids',nargs='*')
    args=parser.parse_args()
    folders=sorted(p.parent.parent.parent for p in ROOT.glob('[0-9][0-9][0-9]_*/original/metadata/metadata.json'))
    for folder in folders:
        if not args.ids or folder.name[:3] in args.ids:
            prepare(folder,args.output)
