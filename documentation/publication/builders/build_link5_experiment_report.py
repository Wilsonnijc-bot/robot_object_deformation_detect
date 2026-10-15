#!/usr/bin/env python3
"""Build a local research report from retained evidence; never run inference."""
from pathlib import Path
import sys, json, gzip, hashlib, base64, random
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from robot.analysis.analyze_link5_updated_mask_auroc import labels, metrics
from infrastructure.shared.replay.reconstruct_replay import transform_world_to_camera
from robot.experiments.link5_pair_selection_ablation.refine_v1 import mask_geometry

OUT = ROOT / 'documentation/reports/link5-experiments'
ABL = ROOT / 'robot/experiments/link5_pair_selection_ablation/results/20261007'
OLD = ROOT / 'robot/experiments/link7_tracker_filter_comparison/results/link5-link7-four-way-selected45-20260927'
SIMPLE = ROOT / 'robot/experiments/link5_simple3d/results/simple3d-20261003-run2/link5_pair_visible_anchor'
CASE = 'COSMOS3_0010'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def dump(p, d):
    p.write_text(json.dumps(d, separators=(',', ':'), allow_nan=False))

def inline_png(a):
    ok, b = cv2.imencode('.png', a)
    assert ok
    return 'data:image/png;base64,' + base64.b64encode(b).decode()

def href(p):
    import os
    return Path(os.path.relpath(p, OUT)).as_posix()

def replay_data(path):
    return json.JSONDecoder().raw_decode(path.read_text().split('const data=', 1)[1])[0]

def pair_rationale_crops():
    """Extract actual video pixels in equal-size, mask-centered Link5 crops."""
    case = 'COSMOS2.5_0054'
    frames = [0, 23, 46, 69]
    manifest = json.loads((ABL/'manifest.json').read_text())
    entry = next(e for e in manifest['entries'] if e['video_id'] == case)
    maskpath = ROOT/'robot/experiments/link5_shape_codebook/results/round_four_frame0'/entry['mask']
    assert sha(maskpath) == entry['mask_sha256']
    archive = np.load(maskpath, allow_pickle=False)
    masks = archive['object_masks'][:, archive['object_names'].tolist().index('link5')]
    video = ABL/f'report/playback/{case}.mp4'
    cap = cv2.VideoCapture(str(video))
    assert cap.isOpened()
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width, height = [int(cap.get(p)) for p in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT)]
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == len(masks)
    boxes = []
    for frame in frames:
        mask = cv2.resize(masks[frame].astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
        yy, xx = np.where(mask)
        boxes.append((int(xx.min()), int(yy.min()), int(xx.max())+1, int(yy.max())+1))
    padding = 24
    crop_w = min(width, max(b[2]-b[0] for b in boxes)+2*padding)
    crop_h = min(height, max(b[3]-b[1] for b in boxes)+2*padding)
    crops = []
    for frame, (left, top, right, bottom) in zip(frames, boxes):
        x = max(0, min(width-crop_w, (left+right-crop_w)//2))
        y = max(0, min(height-crop_h, (top+bottom-crop_h)//2))
        assert x <= left and right <= x+crop_w and y <= top and bottom <= y+crop_h
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, pixels = cap.read()
        assert ok
        image = OUT/f'assets/pair-rationale-{case}-frame-{frame:03d}.png'
        assert cv2.imwrite(str(image), pixels[y:y+crop_h, x:x+crop_w])
        crops.append(dict(frame_zero_based=frame, seconds=frame/fps,
                          source_crop_xywh=[x, y, crop_w, crop_h],
                          image=href(image), image_sha256=sha(image)))
    cap.release()
    result = dict(case=case, fps=fps, source_hw=[height, width],
                  crop_policy='equal-size rectangles centered on each saved Link5 mask; 24-pixel minimum padding before image-edge clamping; no resizing or rotation',
                  video=href(video), video_sha256=sha(video),
                  mask=str(maskpath.relative_to(ROOT)), mask_sha256=sha(maskpath), crops=crops)
    dump(OUT/'assets/pair-rationale.json', result)
    return result

def diagram(name, title, height, nodes, edges):
    from html import escape
    pieces = [f'<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="{height}" viewBox="0 0 1080 {height}" role="img" aria-labelledby="title"><title id="title">{escape(title)}</title><defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10" fill="#667c8c"/></marker></defs><rect width="1080" height="{height}" fill="#fbfcfd"/>']
    lookup={n[0]:n for n in nodes}
    for edge in edges:
        start,end,label=edge[:3]
        _,x,y,w,h,*_ = lookup[start]; _,xx,yy,ww,hh,*_=lookup[end]
        a,b=x+w/2,y+h; c,d=xx+ww/2,yy
        if d < b:
            a,b=x+w,y+h/2;c,d=xx,yy+hh/2
            path=f'M{a},{b} H{(a+c)/2} V{d} H{c}'
        else:
            mid=(b+d)/2;path=f'M{a},{b} V{mid} H{c} V{d}'
        if len(edge)>3:path=edge[3]
        pieces.append(f'<path d="{path}" fill="none" stroke="#667c8c" stroke-width="2" marker-end="url(#arrow)"/>')
        if label:
            tx=(a+c)/2;ty=(b+d)/2-7
            pieces.append(f'<text x="{tx}" y="{ty}" text-anchor="middle" font-size="13" font-family="system-ui,sans-serif" fill="#415b6c" paint-order="stroke" stroke="#fbfcfd" stroke-width="7">{escape(label)}</text>')
    refined_nodes = {
        'robot-architecture.svg': {'l5', 'l7', 'd', 'p5'},
        'object-coordination.svg': {'l', 'repair'},
        'link7-reseeding.svg': {'v2', 'sam', 'g'},
    }.get(name, set())
    for node_id,x,y,w,h,head,lines,kind in nodes:
        refined = node_id in refined_nodes
        if refined:
            kind = 'refined'
        fill,stroke={'default':('#f0f4f8','#ced9e2'),'blue':('#eaf3fc','#90b6d6'),'green':('#e7f3ed','#89b9a5'),'gate':('#fff4de','#d2b170'),'stop':('#fceeea','#d8a594'),'refined':('#efe9f8','#7955a3')}[kind]
        pieces.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="{stroke}"/><text x="{x+w/2}" y="{y+27}" text-anchor="middle" font-family="system-ui,sans-serif" font-size="17" font-weight="650" fill="#172b3b">{escape(head)}</text>')
        if refined:
            pieces.append(f'<rect x="{x}" y="{y}" width="5" height="{h}" rx="2" fill="#7955a3"/><text x="{x+9}" y="{y-8}" font-family="system-ui,sans-serif" font-size="11" font-weight="700" letter-spacing="1" fill="#68418f">REFINED</text>')
        for i,line in enumerate(lines):
            pieces.append(f'<text x="{x+w/2}" y="{y+51+21*i}" text-anchor="middle" font-family="system-ui,sans-serif" font-size="14" fill="#435c6d">{escape(line)}</text>')
    pieces.append('</svg>')
    (OUT/'assets'/name).write_text(''.join(pieces))

def flowcharts():
    diagram('robot-architecture.svg','Robot deformation: masks, geometry, tracking and fixed pair scoring',1125,[
        ('video',350,15,380,70,'Source video',['RGB + visual references'],'blue'),
        ('l2',20,135,320,96,'Link2 masking',['DINOv2 box → SAM3','Full-video propagation'],'default'),
        ('l5',380,135,320,96,'Link5 masking',['DINOv2 box → SAM3 → VLM guard','3 positives + 2 negatives'],'green'),
        ('l7',740,135,320,96,'Link7 masking',['SAM3 → VLM1 / VLM2','Optional object-aware VLM3 repair'],'default'),
        ('m',350,285,380,70,'Validated full-video masks',['Named channels + source identity'],'blue'),
        ('g',25,395,315,115,'MegaSAM geometry',['Depth-Anything + UniDepth','DROID + RAFT → CVD','Depth + poses → world pointmaps'],'default'),
        ('q',385,395,315,115,'Frame-zero point queries',['SIFT + Shi–Tomasi + grid','Spatially balanced queries','Stable point IDs'],'default'),
        ('c',385,560,315,95,'CoTracker3 exact-group',['2D trajectories + visibility','Link5: frozen queries, raw tracks'],'default'),
        ('x',385,700,315,70,'Lift tracks to world XYZ',['Sample geometry at tracked pixels'],'blue'),
        ('d',740,395,315,115,'Link5 all-frame depth support',['2 source-pixel elliptical erosion','32-neighbor normalized density','≥256 pixels + ≥50% retention'],'green'),
        ('p27',20,835,320,115,'Link2 / Link7 baseline pairs',['Frame-zero V1 anchor filter','Rank: 3D baseline × interior distance','Highest 30 ranks; fixed graph'],'default'),
        ('p5',740,835,320,115,'Link5 refine_v1 pairs',['Frame-zero visibility AND support','16 transition-focused + 14 others','Degree ≤3; later support gates use'],'green'),
        ('r',360,990,360,100,'Rigidity score + replay',['Distance ratios → MAD / median','<3 pairs: carry previous score','Mean over frames 1 … T−1'],'blue'),
    ],[('video','l2',''),('video','l5',''),('video','l7',''),('l2','m',''),('l5','m',''),('l7','m',''),('m','q',''),('m','d',''),('video','g','','M350,50 H10 V370 H182 V395'),('q','c',''),('g','x','','M182,510 V735 H385'),('g','d','','M340,452 H365 V535 H897 V510'),('c','x',''),('x','p27',''),('x','p5',''),('d','p5','Support on every frame'),('p27','r',''),('p5','r','')])
    diagram('object-coordination.svg','Link7 and object coordination: final mask handoff and independent scoring branches',1390,[
        ('v',350,15,380,70,'Source video + task prompt',['Same source hash and frame order'],'blue'),
        ('l',30,130,420,100,'Persistent Link7 masks',['Initial SAM3 → surge candidates → VLM1','VLM2 at diagnosed frame + 1','Fresh bidirectional SAM3'],'default'),
        ('o',630,130,420,100,'Task-object masks',['Ground task object at frame zero','Independent SAM3 propagation','Object identity preserved'],'default'),
        ('j',330,290,420,70,'Join and verify both mask archives',['Frame grid, names, dimensions, hashes'],'blue'),
        ('gate',300,410,480,95,'Any frame triggers VLM3?',['Object covered by Link7 >95%','OR Link7 / image area ≥25%'],'gate'),
        ('repair',20,570,470,120,'VLM3 repair at earliest gate frame',['Exact frame; no +1','3 positives + 3 negatives, including object','Fresh bidirectional SAM3 + acceptance audit'],'green'),
        ('core',650,590,380,75,'Keep validated core mask',['No gate hit'],'default'),
        ('commit',350,750,380,95,'Commit final selected Link7 mask',['Mask paths + hashes + status','Allowed crop-frame exclusions'],'blue'),
        ('stop',30,750,260,95,'Required repair fails',['Exhausted coordinator budget','Disable case; save evidence'],'stop'),
        ('robot',30,910,440,100,'Robot branch',['Selected Link7 + shared MegaSAM','Point initialization → CoTracker','V1 anchors → fixed pairs → rigidity'],'default'),
        ('occ',600,910,440,100,'Object occlusion branch',['Object masks + selected Link7 mask','Object CoTracker tracks + visibility','Expected silhouette + temporal loss flags'],'green'),
        ('rr',30,1080,440,75,'Robot score and interactive replay',['Unavailable measurements remain explicit'],'default'),
        ('crop',600,1070,440,120,'Available pixels → reference mapping',['A(t) = object(t) AND NOT Link7(t)','Crop gates + eligible recovery successors','Ten pairs; quotas 0 / 2 / 2 / 2 / 4','Earlier-interval fallback; shared canvases'],'green'),
        ('ad',600,1250,440,95,'AnomalyDINO + crop / occlusion replays',['Identical selected pair PNGs feed scorer','Object geometry is a separate experiment'],'blue'),
    ],[('v','l',''),('v','o',''),('l','j',''),('o','j',''),('j','gate',''),('gate','repair','Yes'),('gate','core','No'),('repair','commit','Accepted'),('repair','stop','Failure'),('core','commit',''),('commit','robot','Same selected mask'),('commit','occ','Same selected mask'),('o','occ','','M1050,180 H1069 V960 H1040'),('robot','rr',''),('occ','crop',''),('crop','ad','')])
    diagram('link7-reseeding.svg','Link7 persistent masking and reseeding',890,[
        ('start',330,15,420,75,'Frame-zero SAM3 initialization',['DINOv2 gripper box; follow selected ID'],'default'),
        ('s',330,145,420,95,'Nominate VLM1 candidate frames',['Upward log-area surge ≥0.25','Review event−4 through event+5','No surge: 30%, 48%, 52%, 60%, 70%'],'default'),
        ('v1',330,295,420,75,'VLM1 diagnoses palm deformation',['Compare candidate RGB crops to reference'],'blue'),
        ('v2',330,425,420,100,'VLM2 at earliest confirmed frame + 1',['3 gripper positives; 2 arm/wrist negatives','Retries use the same selected frame','Invalid required points stop correction'],'green'),
        ('sam',330,585,420,75,'Fresh bidirectional SAM3',['Validate frame lineage, point roles, masks'],'default'),
        ('g',330,715,420,130,'Optional object-aware VLM3',['Requires task-object masks','>95% object overlap OR ≥25% image area','3 positives + 3 negatives at exact gate frame','Accepted mask feeds both downstream paths'],'blue'),
    ],[('start','s',''),('s','v1',''),('v1','v2','Confirmed deformation'),('v2','sam',''),('sam','g','')])

def main():
    (OUT/'assets').mkdir(parents=True, exist_ok=True)
    analysis = json.loads((ABL/'analysis/forearm_AB_correlation.json').read_text())
    selection = json.loads((OLD/'selection.json').read_text())
    workbook = ROOT/'documentation/data/labels/selected_45_matched_videos_styledv2.xlsx'
    assert sha(workbook) == analysis['label_source']['sha256']
    lbl = labels(workbook, selection, 'AB')
    old_scores = json.loads((OLD/'summary.json').read_text())['cases']
    binary_ids = {r['video_id'] for r in analysis['rows'] if r['forearm_AB'] in (0, 1)}
    for row in analysis['rows']:
        if row['forearm_AB'] is not None:
            assert row['forearm_AB'] == lbl[row['video_id']]
    old_points = [(c, lbl[c], r['link5']['epsilon_rigidity']) for c,r in old_scores.items()]
    full_old = metrics(old_points)
    matched_old = metrics([p for p in old_points if p[0] in binary_ids])
    assert abs(full_old['auroc'] - .6695804195804196) < 1e-12
    rows = analysis['rows']
    report = dict(
        case=CASE, rows=rows,
        auroc=dict(previous=full_old, previous_matched=matched_old,
                   current=analysis['statistics']['exact_0_1']['all_labeled'],
                   sensitivity={p: analysis['statistics'][p]['all_labeled'] for p in ('half_as_0','half_as_1')}),
        paths=dict(ablation=href(ABL), previous=href(OLD), simple=href(SIMPLE)),
    )
    with gzip.open(ABL/f'replay_context/{CASE}.json.gz', 'rt') as f:
        ctx = json.load(f)
    trajpath = ABL/f'rigidity/{CASE}/refine_v1/trajectories.npz'
    z = np.load(trajpath, allow_pickle=False)
    poses = np.asarray(ctx['camera_poses'])
    camera = np.stack([transform_world_to_camera(w,p) for w,p in zip(z['sampled_world_xyz'],poses)])
    finite = np.isfinite(camera).all(2) & (camera[:,:,2] > 0)
    raw = (z['raw_visibility'] > .5) & finite
    good = raw & z['query_depth_support']
    t = int(np.argmax((raw & ~good).sum(1)))
    assert t == 30
    receipt = json.loads((ABL/f'depth_support/{CASE}.json').read_text())
    manifest = json.loads((ABL/'manifest.json').read_text())
    entry = next(e for e in manifest['entries'] if e['video_id'] == CASE)
    maskpath = ROOT/'robot/experiments/link5_shape_codebook/results/round_four_frame0'/entry['mask']
    assert sha(maskpath) == entry['mask_sha256']
    masks_archive = np.load(maskpath, allow_pickle=False)
    masks = masks_archive['object_masks'][:, masks_archive['object_names'].tolist().index('link5')].astype(np.uint8)
    assert sha(ABL/f'depth_support/{CASE}.npz') == receipt['support_sha256']
    dep = np.load(ABL/f'depth_support/{CASE}.npz', allow_pickle=False)
    stages = []
    for mask in masks:
        _,components,areas,_ = cv2.connectedComponentsWithStats(mask,8)
        largest=(components==1+int(np.argmax(areas[1:,cv2.CC_STAT_AREA]))).astype(np.uint8)
        stages.append({
            'original': inline_png(mask*255),
            'depth': inline_png(cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)),borderType=cv2.BORDER_CONSTANT,borderValue=0)*255),
            'query': inline_png(cv2.erode(largest, np.ones((5,5),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)*255),
        })
    reasons = dep['rejection_reasons'][t]
    valid = dep['valid_pixels'][t]
    overlay = np.zeros((*valid.shape,4), np.uint8)
    overlay[valid] = [41,154,111,145]
    overlay[reasons==1] = [164,74,158,230]
    overlay[reasons==2] = [239,166,55,220]
    overlay[reasons==3] = [218,77,65,225]
    k = np.asarray(ctx['intrinsic'])
    intrinsic = k.tolist() if k.shape == (4,) else [k[0,0],k[1,1],k[0,2],k[1,2]]
    oldpage = OLD/f'cases/{CASE}/v1_cotracker3/replay/interactive_exact-group/link5_exact-group.html'
    old = replay_data(oldpage)
    newevidence = json.loads((ABL/f'rigidity/{CASE}/refine_v1/evidence.json').read_text())
    balanced = json.loads((ABL/f'rigidity/{CASE}/balanced_v0/evidence.json').read_text())
    geometry=mask_geometry(cv2.resize(masks[0],(valid.shape[1],valid.shape[0]),interpolation=cv2.INTER_NEAREST))
    c=geometry['transition']; low=float(geometry['low'].min());high=float(geometry['high'].max())
    polygon=[]
    for s,v in [(c-.14,low),(c+.14,low),(c+.14,high),(c-.14,high)]:
        xy=geometry['origin']+np.array([geometry['umin']+s*geometry['length'],v])@geometry['basis'].T
        polygon.append((xy*[masks.shape[2]/valid.shape[1],masks.shape[1]/valid.shape[0]]).tolist())
    method = dict(case=CASE, fps=24, frames=len(camera), source_hw=list(masks.shape[1:]),
                  image_hw=ctx['image_hw'], intrinsic=intrinsic,
                  video=href(ABL/f'report/playback/{CASE}.mp4'), erosion=stages,
                  pair=dict(
                      old=dict(points=old['points'][0], cloud=old['clouds'][0], pairs=old['windows'][0]['selected_pairs'],
                               intrinsic=[old['focal_length'],old['focal_length'],old['image_hw'][1]/2,old['image_hw'][0]/2],image_hw=old['image_hw'],ids=old['point_ids']),
                      new=dict(points=[p.tolist() if ok else None for p,ok in zip(camera[0],good[0])],
                               cloud=ctx['clouds'][0],pairs=newevidence['selected_pairs'],
                               intrinsic=intrinsic,image_hw=ctx['image_hw'],ids=z['point_ids'].tolist()),
                      balanced=dict(pairs=balanced['selected_pairs']),
                      selection=newevidence['geometric_selection'],stats=newevidence['selection_stats'],band_polygon=polygon),
                  depth=dict(frame=t, points=camera[t].tolist(),ids=z['point_ids'].tolist(),
                             raw=raw[t].tolist(),good=good[t].tolist(),
                             density=[float(v) if np.isfinite(v) else None for v in dep['query_density'][t]],
                             overlay=inline_png(cv2.cvtColor(overlay,cv2.COLOR_RGBA2BGRA)),
                             info=receipt['frames'][t],dense_cloud=ctx['clouds'][t]))
    # No invented XYZ: raw depth illustration uses the identical retained tracked samples.
    assert np.isfinite(camera).all()
    (OUT/'assets/method-data.js').write_text('window.METHOD_DATA='+json.dumps(method,separators=(',',':'),allow_nan=False)+';')
    rng = random.Random(20261011)
    available = sorted(p.stem for p in (SIMPLE/'replay/cases').glob('*.js'))
    chosen = []
    for prefix in ('COSMOS2.5','COSMOS3','LVP_ROBOWM'):
        chosen.extend(rng.sample([c for c in available if c.startswith(prefix+'_')],2))
    report['simple_sample'] = chosen
    report['simple_seed'] = 20261011
    report['simple_summary'] = json.loads((SIMPLE/'metadata/summary.json').read_text())
    report['pair_rationale'] = pair_rationale_crops()
    from build_link5_pair_availability import build as build_availability
    availability = build_availability()
    report['pair_availability_summary'] = availability['summary']
    dump(OUT/'assets/report-data.json',report)
    (OUT/'assets/report-data.js').write_text('window.REPORT_DATA='+json.dumps(report,separators=(',',':'),allow_nan=False)+';')
    simple_html = (SIMPLE/'replay/index.html').read_text()
    simple_html = simple_html.replace('<head>', '<head><base href="'+href(SIMPLE/'replay')+'/">',1)
    # The retained run moved from the shared Simple3D owner into the robot experiment.
    # Repair its legacy cross-run links in this display copy only.
    import os
    shared=ROOT/'infrastructure/shared/experimental/simple3d/results/simple3d-20261003-run2'
    simple_html=simple_html.replace('href="../../ANALYSIS.md"','href="'+os.path.relpath(shared/'ANALYSIS.md',SIMPLE/'replay')+'"')
    simple_html=simple_html.replace('"earlier_replay":"../../replay/index.html"','"earlier_replay":'+json.dumps(os.path.relpath(shared/'replay/index.html',SIMPLE/'replay')))
    marker=";load().catch(e=>"
    assert marker in simple_html
    simple_html = simple_html.replace(marker,";document.querySelector('a[href=\"#point-clouds\"]').href=location.href.split('#')[0]+'#point-clouds';const requestedCase=new URLSearchParams(location.search).get('case');if(manifest.cases.some(c=>c.video_id===requestedCase))$('case').value=requestedCase;load().catch(e=>",1)
    (OUT/'simple3d.html').write_text(simple_html)
    provenance = dict(built_from_retained_evidence=True,model_inference=False,published=False,
        pair_availability_data='assets/pair-availability.json',
        pair_rationale=report['pair_rationale'],
        method_case=CASE, depth_frame_zero_based=t,depth_cloud_scope='same retained tracked XYZ samples before/after saved depth-support gating; no raw dense-cloud substitution',
        pair_comparison='historical measured naive baseline graph versus current measured refine_v1 graph; different runs, not a controlled graph-only comparison',
        simple_random_seed=20261011,simple_random_policy='two random videos per generator, sorted identities before seeded sampling',simple_sample=chosen,
        previous_auroc=full_old,previous_matched_auroc=matched_old,
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in [workbook,OLD/'summary.json',OLD/'selection.json',oldpage,ABL/'manifest.json',ABL/'analysis/forearm_AB_correlation.json',trajpath,ABL/f'depth_support/{CASE}.npz',maskpath,SIMPLE/'metadata/summary.json',ROOT/'robot/ARCHITECTURE.md',ROOT/'documentation/link7_object_masking_occlusion_cropping.md']})
    dump(OUT/'assets/provenance.json',provenance)
    flowcharts()
    print(json.dumps(dict(output=str(OUT),previous_auroc=full_old['auroc'],previous_matched=matched_old['auroc'],sample=chosen,depth_frame=t),indent=2))

if __name__ == '__main__':
    main()
