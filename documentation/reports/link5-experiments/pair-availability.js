'use strict';
(() => {
  const A=window.PAIR_AVAILABILITY;
  const byId=id=>document.getElementById(id);
  const el=(tag,text)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;return n;};
  const integer=n=>n.toLocaleString('en-US');
  const names={link2:'Link2 · selected-45 V1',link5:'Link5 · refine_v1'};
  const range=a=>a[0]===a[1]?String(a[0]):a.join('–');
  function summaryRows(id,data){
    for(const link of ['link2','link5']){
      const s=data[link],tr=el('tr');
      const total=el('td',integer(s.scoring_pair_measurements));total.className='numeric scoring-total';
      tr.append(el('th',names[link]),el('td',integer(s.videos)),el('td',range(s.selected_pair_range)),el('td',integer(s.scoring_frames)),total,el('td',s.mean_pairs_per_scored_frame.toFixed(2)),el('td',integer(s.carried_frames)));
      byId(id).append(tr);
    }
  }
  summaryRows('scoring-pair-summary',A.summary);
  summaryRows('scoring-pair-matched',A.matched);
  byId('scoring-matched-count').textContent=A.matched_ids.length;
  function renderCases(){
    const query=byId('scoring-pair-search').value.trim().toLowerCase(),link=byId('scoring-pair-link').value;
    const sort=byId('scoring-pair-sort').value;
    const rows=A.cases.filter(r=>(link==='all'||r.link===link)&&r.video_id.toLowerCase().includes(query));
    rows.sort(sort==='video'?(a,b)=>a.video_id.localeCompare(b.video_id)||a.link.localeCompare(b.link):(a,b)=>(sort==='most'?b.scoring_pair_measurements-a.scoring_pair_measurements:a.scoring_pair_measurements-b.scoring_pair_measurements)||a.video_id.localeCompare(b.video_id));
    byId('scoring-case-rows').replaceChildren();
    for(const r of rows){
      const tr=el('tr'),name=el('th'),a=el('a',r.video_id);
      a.href=r.link==='link2'?'../../../robot/results/selected-45-v1/cases/'+r.video_id+'/v1/replay/interactive_exact-group/link2_exact-group.html':'../../../robot/experiments/link5_pair_selection_ablation/results/20261007/report/cases/'+r.video_id+'/refine_v1.html';
      a.target='_blank';a.rel='noopener';name.append(a);
      if(r.cohort==='additional'){const n=el('div','Additional case');n.className='tiny';name.append(n);}
      const total=el('td',integer(r.scoring_pair_measurements));total.className='numeric';
      tr.append(name,el('td',r.link==='link2'?'Link2':'Link5 refine_v1'),el('td',r.selected_pairs),el('td',integer(r.scoring_frames)),total,el('td',r.mean_pairs_per_scored_frame.toFixed(2)),el('td',r.carried_frames));
      byId('scoring-case-rows').append(tr);
    }
    if(!rows.length){const tr=el('tr'),td=el('td','No cases match this search.');td.colSpan=7;tr.append(td);byId('scoring-case-rows').append(tr);}
    byId('scoring-case-count').textContent=rows.length+' / '+A.cases.length+' link/video records · open a video ID to inspect its saved replay';
  }
  byId('scoring-pair-search').addEventListener('input',renderCases);
  byId('scoring-pair-link').addEventListener('change',renderCases);
  byId('scoring-pair-sort').addEventListener('change',renderCases);
  renderCases();
})();
