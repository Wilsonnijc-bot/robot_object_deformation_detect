#!/usr/bin/env python3
"""Build lossless generation references from per-video visual action reviews.

Original TAVP source files remain unchanged. Dobot derivatives retain only the
middle third-person panel. Re-encoded references use lossless H.264 (CRF 0).
"""
import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FULL_SEQUENCE_PROMPT_FILENAME = 'prompt for original multi step action(reproducing the original video).txt'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(text,encoding='utf-8')
    temp.replace(path)


def write_json(path, data):
    write_text(path,json.dumps(data,indent=2,ensure_ascii=False)+'\n')


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def run(command):
    result=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=600)
    if result.returncode or result.stderr.strip():
        raise RuntimeError(result.stderr.decode(errors='replace')[-3000:])
    return result.stdout


def probe(path):
    return json.loads(run(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
                           '-show_streams','-show_format','-of','json',str(path)]))


def crop_filter(original):
    if original['robot_type'].startswith('Dobot'):
        width=original['resolution']['width']//3
        height=original['resolution']['height']
        return f'crop={width}:{height}:{width}:0', {'x':width,'y':0,'width':width,'height':height}
    return None, None


def frame_pixels(video, frame_index=0, crop=None):
    filters=([crop] if crop else [])+[f'select=eq(n\\,{frame_index})']
    return run(['ffmpeg','-v','error','-xerror','-noautorotate','-i',str(video),
                '-vf',','.join(filters),'-frames:v','1','-pix_fmt','rgb24','-f','rawvideo','-'])


def frame_png(video, image, frame_index=0, crop=None):
    filters=([crop] if crop else [])+[f'select=eq(n\\,{frame_index})']
    temp=image.with_name(image.stem+'.tmp.png')
    run(['ffmpeg','-v','error','-xerror','-y','-noautorotate','-i',str(video),'-vf',','.join(filters),
         '-frames:v','1','-pix_fmt','rgb24','-c:v','png','-threads','1',str(temp)])
    temp.replace(image)


def encode(source, output, count, crop, entire_count):
    if crop is None and count==entire_count:
        shutil.copy2(source,output)
        return 'original_bytes_copied'
    filters=([crop] if crop else [])+[f'trim=start_frame=0:end_frame={count}','setpts=PTS-STARTPTS']
    temp=output.with_name('video.tmp.mp4')
    run(['ffmpeg','-v','error','-xerror','-y','-noautorotate','-i',str(source),'-map','0:v:0',
         '-vf',','.join(filters),'-c:v','libx264','-crf','0','-preset','medium','-pix_fmt','yuv420p',
         '-fps_mode','passthrough','-an','-movflags','+faststart',str(temp)])
    temp.replace(output)
    return 'lossless_h264_crf_0'


def validate_review(review, original):
    count=original['decoded_frame_count']
    assert review['video_id']==original['video_id'], 'Review video identity mismatch'
    assert review['source_frame_count']==count and review['source_fps']==original['fps'], 'Review timeline mismatch'
    assert review['inspected_frame_stride']==2, 'Dense whole-video review is required'
    assert review['inspected_sampled_frame_count']==math.ceil(count/2), 'Incomplete visual coverage'
    expected={f'sheet_{n:03d}.jpg' for n in range(1,math.ceil(math.ceil(count/2)/64)+1)}
    assert set(review['reviewed_contact_sheets'])==expected, 'Unreviewed whole-video sheet'
    assert review['native_detail_frames'], 'Native detail review is required'
    assert all(isinstance(n,int) and 0<=n<count for n in review['native_detail_frames']), 'Invalid detail frame'
    selection=review['selected_segment']
    assert selection['start_frame']==0, 'Generation reference must start at original initial frame'
    assert isinstance(selection['end_frame_exclusive'],int) and 0<selection['end_frame_exclusive']<=count, 'Invalid cut'
    assert review['observed_sequence'] and review['initial_scene'], 'Missing visual observations'
    for event in review['observed_sequence']:
        assert 0<=event['start_frame']<=event['end_frame_inclusive']<count, 'Observed event lies outside source timeline'
    assert review['world_model_prompt'].strip() and review['full_sequence_prompt'].strip(), 'Empty generation prompt'
    assert review['prompt_provenance']=='visually_authored_from_video_and_initial_frame', 'Invalid prompt provenance'
    assert review.get('prompt_protocol')=='concise_action_v1', 'Prompt must follow project concise protocol'
    suffix='Keep the camera fixed; preserve robot geometry, physically valid joint motion, and normal gripper opening and closing.'
    for field in ['world_model_prompt','full_sequence_prompt']:
        assert review[field].endswith(suffix), 'Shared concise physical/camera constraint is required'
    assert review['selected_execution_status'] in {'success','failure','partial','uncertain'}, 'Missing selected action outcome'
    assert all(0<=n<count for n in review['evidence_frame_indices']), 'Invalid evidence frame'


def make_variant(folder, target, original, review, count, prompt, variant, verify_only):
    source=folder/'original/video.mp4'
    crop, rectangle=crop_filter(original)
    video=target/'video.mp4'
    image=target/'first_frame.png'
    prompt_path=target/'prompt.txt'
    metadata_path=target/'metadata/metadata.json'
    if not verify_only:
        target.mkdir(parents=True,exist_ok=True)
        prior=read_json(metadata_path) if metadata_path.exists() else {}
        reusable=(video.exists() and prior.get('source_video_sha256')==original['sha256']
                  and prior.get('source_frame_end_exclusive')==count and prior.get('crop_rectangle')==rectangle
                  and prior.get('video_sha256')==digest(video))
        encoding=prior['encoding'] if reusable else encode(source,video,count,crop,original['decoded_frame_count'])
        frame_png(source,image,crop=crop)
        write_text(prompt_path,prompt.strip()+'\n')
    else:
        encoding=read_json(metadata_path)['encoding']
    info=probe(video)
    stream=info['streams'][0]
    expected_width=rectangle['width'] if rectangle else original['resolution']['width']
    expected_height=original['resolution']['height']
    assert (stream['width'],stream['height'])==(expected_width,expected_height), 'Incorrect derivative dimensions'
    image_stream=probe(image)['streams'][0]
    assert image_stream['codec_name']=='png' and (image_stream['width'],image_stream['height'])==(expected_width,expected_height), 'Initial image is not PNG at source-view resolution'
    assert int(stream['nb_read_frames'])==count, 'Derivative frame count differs from requested exact cut'
    assert float(eval_fraction(stream['avg_frame_rate']))==original['fps'], 'FPS changed'
    run(['ffmpeg','-v','error','-xerror','-err_detect','explode','-noautorotate','-i',str(video),'-map','0:v:0','-f','null','-'])
    source_first=frame_pixels(source,crop=crop)
    assert frame_pixels(video)==source_first, 'Lossless derivative initial pixels differ from cropped original'
    assert frame_pixels(image)==source_first, 'Initial PNG differs from cropped source frame zero'
    assert frame_pixels(video,count-1)==frame_pixels(source,count-1,crop=crop), 'Last retained frame differs from source frame'
    assert prompt_path.read_text(encoding='utf-8').strip()==prompt.strip(), 'Prompt artifact mismatch'
    metadata={
        'dataset':'TAVP','video_id':original['video_id'],'robot_type':original['robot_type'],
        'variant':variant,'generation_prompt':prompt.strip(),'prompt_provenance':review['prompt_provenance'],
        'prompt_protocol':review['prompt_protocol'],
        'original_task_name':original['task_name'],'original_execution_status':original['execution_status'],
        'selected_execution_status':review['selected_execution_status'] if variant=='first_substep' or count==review['selected_segment']['end_frame_exclusive'] else None,
        'original_annotation_provenance':original['prompt_provenance'],
        'source_video':str(source.relative_to(ROOT)),'source_video_sha256':original['sha256'],
        'source_url':original['original_video_url'],'source_repository_commit':original['repository_commit'],
        'view':'middle_third_person' if rectangle else 'original_single_camera',
        'crop_rectangle':rectangle,'spatial_resize':False,'encoding':encoding,
        'source_frame_start':0,'source_frame_end_exclusive':count,
        'source_frame_count':original['decoded_frame_count'],'frame_count':count,
        'first_frame_index':0,'fps':original['fps'],'fps_rational':stream['avg_frame_rate'],
        'resolution':{'width':expected_width,'height':expected_height},
        'duration_seconds':float(info['format']['duration']),'codec':stream['codec_name'],
        'file_size_bytes':video.stat().st_size,'video_sha256':digest(video),
        'first_frame_sha256':digest(image),'first_frame_rgb24_sha256':hashlib.sha256(source_first).hexdigest(),
        'observed_terminal_state':review['selected_segment']['terminal_state'] if variant=='first_substep' else review['observed_sequence'][-1]['action'],
        'selection_reason':review['selected_segment']['reason'] if variant=='first_substep' else 'Complete original filmed sequence in the generation view',
        'uncertainties':review['uncertainties'],'manual_annotation_review_needed':bool(review['uncertainties']),
        'action_review_path':str((folder/'generation input/metadata/action_review.json').relative_to(ROOT)),
        'relative_video_path':str(video.relative_to(ROOT)),
        'relative_first_frame_path':str(image.relative_to(ROOT)),
        'relative_prompt_path':str(prompt_path.relative_to(ROOT)),
        'relative_metadata_path':str(metadata_path.relative_to(ROOT)),
        'validation':{'full_decode':True,'frame_count_exact':True,'fps_preserved':True,
                      'dimensions_correct':True,'source_first_frame_pixels_exact':True,
                      'source_last_retained_frame_pixels_exact':True,'png_first_frame_pixels_exact':True,
                      'prompt_nonempty':True},
        'validation_status':'passed',
    }
    if verify_only:
        assert read_json(metadata_path)==metadata,'Saved derivative metadata differs from current validation'
    else:
        write_json(metadata_path,metadata)
    return metadata


def eval_fraction(value):
    numerator,denominator=value.split('/')
    return int(numerator)/int(denominator)


def process(folder, verify_only):
    original=read_json(folder/'original/metadata/metadata.json')
    review=read_json(folder/'generation input/metadata/action_review.json')
    validate_review(review,original)
    assert digest(folder/'original/video.mp4')==original['sha256'],'Original MP4 integrity failed'
    target=folder/'generation input'
    selected_count=review['selected_segment']['end_frame_exclusive']
    full_prompt=target/(FULL_SEQUENCE_PROMPT_FILENAME if selected_count<original['decoded_frame_count'] else 'prompt.txt')
    if selected_count<original['decoded_frame_count']:
        selected=make_variant(folder,target/'first_substep',original,review,selected_count,review['world_model_prompt'],'first_substep',verify_only)
        if verify_only:
            assert full_prompt.read_text(encoding='utf-8').strip()==review['full_sequence_prompt'].strip(), 'Full-sequence prompt mismatch'
        else:
            write_text(full_prompt,review['full_sequence_prompt'].strip()+'\n')
        # Keep the selected reference plus full-sequence text; omit redundant full media.
        for name in ['video.mp4','first_frame.png','metadata/metadata.json']:
            redundant=target/name
            if verify_only:
                assert not redundant.exists(), f'Redundant full-reference artifact: {redundant}'
            elif redundant.exists():
                redundant.unlink()
    else:
        assert review['world_model_prompt'].strip()==review['full_sequence_prompt'].strip(),'Unsimplified prompts must agree'
        selected=make_variant(folder,target,original,review,selected_count,review['world_model_prompt'],'full_sequence',verify_only)
    print(f"[{original['video_id']}] validated {selected['variant']}: {selected_count}/{original['decoded_frame_count']} frames, {selected['view']}",flush=True)
    return {'selected':selected,'review':review,'original':original,
            'full_reference_prompt_path':str(full_prompt.relative_to(ROOT))}


FIELDS=['video_id','robot_type','original_task_name','original_execution_status','selected_execution_status','variant','view','generation_prompt','prompt_protocol',
        'prompt_provenance','relative_video_path','relative_first_frame_path','relative_prompt_path','relative_metadata_path',
        'full_reference_prompt_path',
        'action_review_path','source_video','source_frame_start','source_frame_end_exclusive','source_frame_count',
        'frame_count','fps','duration_seconds','width','height','file_size_bytes','video_sha256','first_frame_sha256',
        'manual_annotation_review_needed','validation_status']


def row(data):
    result={key:data.get(key) for key in FIELDS}
    result.update(width=data['resolution']['width'],height=data['resolution']['height'])
    result['manual_annotation_review_needed']=str(data['manual_annotation_review_needed']).lower()
    return {key:'' if value is None else str(value) for key,value in result.items()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ids',nargs='*',help='Build only these reviewed video IDs; final validation requires all 23')
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    folders=sorted(p.parent.parent.parent for p in ROOT.glob('[0-9][0-9][0-9]_*/original/metadata/metadata.json'))
    if args.ids:
        folders=[p for p in folders if p.name[:3] in args.ids]
    results=[]
    failures=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        tasks={pool.submit(process,folder,args.verify_only):folder for folder in folders}
        for task in concurrent.futures.as_completed(tasks):
            try:
                results.append(task.result())
            except Exception as error:
                failures.append({'video_id':tasks[task].name[:3],'error':str(error)})
                print(f"FAILED {tasks[task].name}: {error}",flush=True)
    results.sort(key=lambda result:result['selected']['video_id'])
    if args.ids:
        if failures:
            raise SystemExit(1)
        return
    expected_rows=[]
    for result in results:
        item=row(result['selected'])
        item['full_reference_prompt_path']=result['full_reference_prompt_path']
        expected_rows.append(item)
    manifest=ROOT/'world_model_manifest.csv'
    if args.verify_only:
        try:
            with manifest.open(encoding='utf-8',newline='') as f:
                assert list(csv.DictReader(f))==expected_rows,'Generation manifest does not match validated references'
        except Exception as error:
            failures.append({'error':str(error)})
    else:
        temp=manifest.with_name(manifest.name+'.tmp')
        with temp.open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(expected_rows)
        temp.replace(manifest)
    summary={
        'complete':len(results)==23 and not failures,'expected_videos':23,'validated_videos':len(results),
        'dedicated_video_review_agents':len(results),
        'robot_counts':dict(Counter(r['selected']['robot_type'] for r in results)),
        'selected_variant_counts':dict(Counter(r['selected']['variant'] for r in results)),
        'selected_execution_status_counts':dict(Counter(r['selected']['selected_execution_status'] for r in results)),
        'selected_view_counts':dict(Counter(r['selected']['view'] for r in results)),
        'selected_resolution_fps_counts':dict(Counter(f"{r['selected']['resolution']['width']}x{r['selected']['resolution']['height']} @ {r['selected']['fps']:g}" for r in results)),
        'visually_inspected_sampled_frames':sum(r['review']['inspected_sampled_frame_count'] for r in results),
        'visually_inspected_native_detail_frames':sum(len(r['review']['native_detail_frames']) for r in results),
        'selected_video_total_bytes':sum(r['selected']['file_size_bytes'] for r in results),
        'selected_source_seconds':sum(r['selected']['duration_seconds'] for r in results),
        'original_source_seconds':sum(r['original']['decoded_frame_count']/r['original']['fps'] for r in results),
        'annotation_uncertainties':[{'video_id':r['selected']['video_id'],'uncertainties':r['review']['uncertainties']} for r in results if r['review']['uncertainties']],
        'failures':failures,'validation_time_utc':datetime.now(timezone.utc).isoformat(),
        'validation_mode':'verify_only' if args.verify_only else 'build_and_validate',
    }
    write_json(ROOT/'world_model_validation_report.json',summary)
    print(json.dumps(summary,indent=2),flush=True)
    if not summary['complete']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
