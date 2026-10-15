#!/usr/bin/env python3
"""Summarize retained CoTracker pair evidence. No inference or source edits."""
from pathlib import Path
import csv, hashlib, json
from collections import Counter
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'documentation/reports/link5-experiments/assets'
L2 = ROOT/'robot/results/selected-45-v1'
L5 = ROOT/'robot/experiments/link5_pair_selection_ablation/results/20261007'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def source(path):
    return {'path':str(path.relative_to(ROOT)), 'sha256':sha(path)}

def case_metrics(case, link, cohort, evidence, available, raw, evidence_path, extra_sources=()):
    pairs=evidence['selected_pairs']; total=len(pairs)
    counts=np.asarray(available, dtype=int)[1:]; raw=np.asarray(raw, dtype=int)[1:]
    assert len(counts)==len(evidence['rigidity_history'])-1 and len(counts)>0
    assert np.all((0<=counts)&(counts<=raw)&(raw<=total))
    carried=set(evidence['carried_frames'])
    assert carried==set((np.flatnonzero(counts<3)+1).tolist())
    records={r['frame']:r for r in evidence['pair_frames']}
    assert set(records)==set(range(1,len(counts)+1))-carried
    for t,r in records.items():
        assert len(r['pair_indices'])==counts[t-1]
    used_pairs = sum(len(r['pair_indices']) for r in records.values())
    degrees=Counter(i for p in pairs for i in (p['track_i'],p['track_j']))
    run=maxrun=0
    for value in counts<3:
        run=run+1 if value else 0;maxrun=max(maxrun,run)
    return dict(video_id=case,generator=case.rsplit('_',1)[0],link=link,cohort=cohort,
                selected_pairs=total,frames=len(counts),available_pairs_sum=int(counts.sum()),
                scoring_pair_measurements=used_pairs,scoring_frames=len(records),
                mean_pairs_per_scored_frame=used_pairs/len(records),
                raw_visible_pairs_sum=int(raw.sum()),pair_slots=total*len(counts),
                mean_available_pairs=float(counts.mean()),availability=float(counts.mean()/total),
                raw_visibility=float(raw.mean()/total),p10_available_pairs=float(np.percentile(counts,10)),
                minimum_available_pairs=int(counts.min()),zero_pair_frames=int((counts==0).sum()),
                full_pair_frames=int((counts==total).sum()),carried_frames=len(carried),
                fresh_score_fraction=float((counts>=3).mean()),longest_carry_frames=maxrun,
                distinct_endpoints=len(degrees),max_endpoint_degree=max(degrees.values()),
                availability_history=(counts/total).tolist(),raw_visibility_history=(raw/total).tolist(),
                sources=[source(evidence_path),*[source(p) for p in extra_sources]])

def aggregate(rows):
    slots=sum(r['pair_slots'] for r in rows);frames=sum(r['frames'] for r in rows)
    history=[v for r in rows for v in r['availability_history']]
    return dict(videos=len(rows),frames=frames,pair_slots=slots,
                scoring_pair_measurements=sum(r['scoring_pair_measurements'] for r in rows),
                scoring_frames=sum(r['scoring_frames'] for r in rows),
                mean_pairs_per_scored_frame=sum(r['scoring_pair_measurements'] for r in rows)/sum(r['scoring_frames'] for r in rows),
                selected_graph_pairs=sum(r['selected_pairs'] for r in rows),
                available_pairs_sum=sum(r['available_pairs_sum'] for r in rows),
                mean_available_pairs=sum(r['available_pairs_sum'] for r in rows)/frames,
                frame_weighted_availability=sum(r['available_pairs_sum'] for r in rows)/slots,
                video_mean_availability=float(np.mean([r['availability'] for r in rows])),
                p10_frame_availability=float(np.percentile(history,10)),
                fresh_score_fraction=1-sum(r['carried_frames'] for r in rows)/frames,
                carried_frames=sum(r['carried_frames'] for r in rows),
                videos_with_carry=sum(r['carried_frames']>0 for r in rows),
                full_pair_fraction=sum(r['full_pair_frames'] for r in rows)/frames,
                zero_pair_fraction=sum(r['zero_pair_frames'] for r in rows)/frames,
                raw_pair_visibility=sum(r['raw_visible_pairs_sum'] for r in rows)/slots,
                support_loss_pp=100*sum(r['raw_visible_pairs_sum']-r['available_pairs_sum'] for r in rows)/slots,
                mean_distinct_endpoints=float(np.mean([r['distinct_endpoints'] for r in rows])),
                mean_max_endpoint_degree=float(np.mean([r['max_endpoint_degree'] for r in rows])),
                selected_pair_range=[min(r['selected_pairs'] for r in rows),max(r['selected_pairs'] for r in rows)])

def build():
    rows=[]
    selection=json.loads((L2/'selection.json').read_text())
    ids=sorted(v['job_id'].rsplit('-',1)[0] for v in selection['videos'])
    missing=[]
    for case in ids:
        p=L2/f'cases/{case}/v1/replay/interactive_exact-group/link2_exact-group_pairs.json'
        if not p.exists():
            missing.append(case);continue
        e=json.loads(p.read_text());assert e['object']=='link2' and e['version']=='v1'
        archive=L2/f'cases/{case}/v1/cotracker_exact-group.npz'
        metrics=L2/f'cases/{case}/v1/metrics.json'
        assert sha(metrics)==e['source_metrics_sha256']
        z=np.load(archive,allow_pickle=False);index=z['object_names'].tolist().index('link2')
        offsets=z['object_offsets'];v=z['visibility'][:,offsets[index]:offsets[index+1]]>.5
        raw=np.array([(v[:,p['track_i']]&v[:,p['track_j']]) for p in e['selected_pairs']]).sum(0)
        rows.append(case_metrics(case,'link2','selected45',e,raw,raw,p,(archive,metrics)))
    manifest=json.loads((L5/'manifest.json').read_text())
    for entry in manifest['entries']:
        case=entry['video_id'];p=L5/f'rigidity/{case}/refine_v1/evidence.json'
        e=json.loads(p.read_text());assert e['pair_method']=='refine_v1'
        frames=e['frame_diagnostics']
        assert [f['frame'] for f in frames]==list(range(len(frames)))
        rows.append(case_metrics(case,'link5',entry['cohort'],e,
                                [f['available_pair_count'] for f in frames],
                                [f['raw_visible_pair_count'] for f in frames],p))
    ids2={r['video_id'] for r in rows if r['link']=='link2'}
    ids5={r['video_id'] for r in rows if r['link']=='link5' and r['cohort']=='selected45'}
    matched=sorted(ids2&ids5)
    aggregates={link:aggregate([r for r in rows if r['link']==link]) for link in ['link2','link5']}
    matched_aggregates={link:aggregate([r for r in rows if r['link']==link and r['video_id'] in matched]) for link in ['link2','link5']}
    generators={g:{link:aggregate([r for r in rows if r['link']==link and r['generator']==g]) for link in ['link2','link5']} for g in ['COSMOS2.5','COSMOS3','LVP_ROBOWM']}
    result=dict(frame_policy='Exclude frame zero; every later frame contributes, including carried-score frames',
                final_scoring_count_policy='Sum len(pair_indices) from saved pair_frames; exclude frame zero and all carried frames. Each pair reused at another frame is another pair measurement, not another unique selected graph edge.',
                availability_definition='Available selected pairs / selected frame-zero pairs; pooled pair-frame weighting',
                link2_policy='Both endpoints visible >0.5 under saved selected-45 V1 exact-group CoTracker; no all-frame depth-support gate',
                link5_policy='Saved refine_v1 available_pair_count: both endpoints visible >0.5 AND all-frame depth support',
                link2_selected=45,link2_unscored=missing,link5_primary=40,link5_additional=1,
                matched_ids=matched,summary=aggregates,matched=matched_aggregates,generators=generators,
                cases=sorted(rows,key=lambda r:(r['video_id'],r['link'])),
                provenance=[source(L2/'selection.json'),source(L2/'summary.json'),source(L5/'manifest.json')])
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'pair-availability.json').write_text(json.dumps(result,separators=(',',':'),allow_nan=False))
    (OUT/'pair-availability.js').write_text('window.PAIR_AVAILABILITY='+json.dumps(result,separators=(',',':'),allow_nan=False)+';')
    fields=[k for k in rows[0] if k not in ['availability_history','raw_visibility_history','sources']]
    with (OUT/'pair-availability.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        writer.writerows({k:r[k] for k in fields} for r in result['cases'])
    return result

if __name__=='__main__':
    data=build();print(json.dumps({'summary':data['summary'],'matched_videos':len(data['matched_ids']),'unscored':data['link2_unscored']},indent=2))
