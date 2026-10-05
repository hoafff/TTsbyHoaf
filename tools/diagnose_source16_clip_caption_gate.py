from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / 'outputs' / 'source16_keep_auto_pilot_review' / 'REVIEW_RESULTS.csv'
PILOT = ROOT / 'kaggle_output' / 'source16-keep-auto-cut-pilot' / 'source16_keep_auto_cut_pilot' / 'outputs' / 'LISTENING_INDEX.csv'
WHISPER_ROOT = ROOT / 'kaggle_output' / 'source16-whisper-consensus'
OUT = ROOT / 'outputs' / 'source16_keep_auto_caption_gate_diagnostic'

def norm(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'[^0-9a-zA-ZÀ-ỹĐđ\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def sim(a: str, b: str) -> float:
    aa, bb = norm(a).split(), norm(b).split()
    if not aa and not bb:
        return 1.0
    if not aa or not bb:
        return 0.0
    return SequenceMatcher(None, aa, bb, autojunk=False).ratio()

def find_caption_events(video_id: str) -> Path:
    candidates = []
    for p in WHISPER_ROOT.rglob('YOUTUBE_CAPTION_EVENTS.csv'):
        if video_id in str(p.parent):
            candidates.append(p)
    if not candidates:
        raise FileNotFoundError(f'Caption events not found for {video_id} under {WHISPER_ROOT}')
    return sorted(candidates)[0]

def best_caption_match(events: pd.DataFrame, start: float, end: float, text: str):
    s = pd.to_numeric(events['start_sec'], errors='coerce')
    e = pd.to_numeric(events['end_sec'], errors='coerce')
    near = events[(e >= start - 3.0) & (s <= end + 3.0)].copy()
    if near.empty:
        return 0.0, '', None, None, 0
    near = near.sort_values(['start_sec','end_sec'], kind='stable').reset_index(drop=True)
    best = (-1.0, '', None, None, 0)
    n = len(near)
    for i in range(n):
        parts=[]
        for j in range(i, min(n, i+12)):
            parts.append(str(near.iloc[j]['text']))
            cand=' '.join(parts).strip()
            score=sim(text,cand)
            if score > best[0]:
                best=(score,cand,float(near.iloc[i]['start_sec']),float(near.iloc[j]['end_sec']),j-i+1)
    return best

def main() -> int:
    if not REVIEW.exists() or not PILOT.exists():
        raise SystemExit('Missing review or pilot files.')
    review=pd.read_csv(REVIEW,low_memory=False).fillna('')
    review=review[review['verdict'].astype(str).str.len()>0].copy()
    pilot=pd.read_csv(PILOT,low_memory=False)
    merged=review.merge(pilot[['clip_id','requested_start_sec','requested_end_sec','text']],on='clip_id',how='left',suffixes=('_review','_pilot'))
    cache={}
    rows=[]
    for r in merged.itertuples(index=False):
        vid=str(r.video_id)
        if vid not in cache:
            cache[vid]=pd.read_csv(find_caption_events(vid),low_memory=False)
        score,cap,cstart,cend,count=best_caption_match(cache[vid],float(r.requested_start_sec),float(r.requested_end_sec),str(r.text_pilot))
        rows.append({
            'review_order':int(r.review_order),
            'video_id':vid,
            'clip_id':str(r.clip_id),
            'verdict':str(r.verdict),
            'pilot_text':str(r.text_pilot),
            'best_caption_similarity':float(score),
            'best_caption_text':cap,
            'caption_start_sec':cstart,
            'caption_end_sec':cend,
            'caption_event_count':int(count),
        })
    out=pd.DataFrame(rows).sort_values('review_order')
    OUT.mkdir(parents=True,exist_ok=True)
    out.to_csv(OUT/'REVIEWED_CLIP_CAPTION_ALIGNMENT.csv',index=False,encoding='utf-8-sig')

    stats={}
    for verdict,g in out.groupby('verdict'):
        s=pd.to_numeric(g['best_caption_similarity'],errors='coerce').dropna()
        stats[str(verdict)]={
            'count':int(len(g)),
            'mean':float(s.mean()) if len(s) else None,
            'p10':float(s.quantile(0.10)) if len(s) else None,
            'p50':float(s.quantile(0.50)) if len(s) else None,
            'p90':float(s.quantile(0.90)) if len(s) else None,
        }
    summary={
        'stage':'SOURCE16_KEEP_AUTO_CLIP_CAPTION_GATE_DIAGNOSTIC_V1',
        'reviewed_rows':int(len(out)),
        'by_verdict':stats,
        'bad_text_below_0_85':int(((out['verdict'].isin(['BAD_TEXT','BAD_BOTH'])) & (out['best_caption_similarity']<0.85)).sum()),
        'good_below_0_85':int(((out['verdict']=='GOOD') & (out['best_caption_similarity']<0.85)).sum()),
        'interpretation':'If BAD_TEXT is mostly low-similarity while GOOD is mostly high-similarity, clip-local caption gating can likely rescue KEEP_AUTO. If both groups look similar, captions are not reliable ground truth for this failure mode.'
    }
    (OUT/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    print('\n===== REVIEWED CLIP ALIGNMENT =====')
    print(out[['review_order','verdict','video_id','clip_id','best_caption_similarity','pilot_text','best_caption_text']].to_string(index=False))
    print('\nOUTPUT',OUT)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
