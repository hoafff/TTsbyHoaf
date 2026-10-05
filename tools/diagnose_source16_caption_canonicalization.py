from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
REVIEW=ROOT/'outputs'/'source16_keep_auto_pilot_review'/'REVIEW_RESULTS.csv'
PILOT=ROOT/'kaggle_output'/'source16-keep-auto-cut-pilot'/'source16_keep_auto_cut_pilot'/'outputs'/'LISTENING_INDEX.csv'
WHISPER_ROOT=ROOT/'kaggle_output'/'source16-whisper-consensus'
OUT=ROOT/'outputs'/'source16_keep_auto_caption_canonicalization'

WORD_RE=re.compile(r'[0-9A-Za-zÀ-ỹĐđ]+')

def toks(s:str):
    return WORD_RE.findall(str(s).lower())

def find_events(video_id:str)->Path:
    ms=[]
    for p in WHISPER_ROOT.rglob('YOUTUBE_CAPTION_EVENTS.csv'):
        if video_id in str(p.parent): ms.append(p)
    if not ms: raise FileNotFoundError(video_id)
    return sorted(ms)[0]

def best_subseq(whisper_text:str, caption_text:str):
    w=toks(whisper_text); c=toks(caption_text)
    if not w or not c: return 0.0,''
    m=len(w); n=len(c)
    lo=max(1,m-max(4,int(m*0.25))); hi=min(n,m+max(6,int(m*0.35)))
    best=(-1.0,0,0)
    for L in range(lo,hi+1):
        for i in range(0,n-L+1):
            cand=c[i:i+L]
            ratio=SequenceMatcher(None,w,cand,autojunk=False).ratio()
            length_pen=0.03*abs(L-m)/max(1,m)
            score=ratio-length_pen
            if score>best[0]: best=(score,i,i+L)
    score,i,j=best
    cand=c[i:j]
    ratio=SequenceMatcher(None,w,cand,autojunk=False).ratio()
    return float(ratio),' '.join(cand)

def main():
    review=pd.read_csv(REVIEW,low_memory=False).fillna('')
    review=review[review['verdict'].astype(str).str.len()>0].copy()
    pilot=pd.read_csv(PILOT,low_memory=False)
    merged=review.merge(pilot[['clip_id','requested_start_sec','requested_end_sec','text']],on='clip_id',how='left',suffixes=('_review','_pilot'))
    cache={}; rows=[]
    for r in merged.itertuples(index=False):
        vid=str(r.video_id)
        if vid not in cache: cache[vid]=pd.read_csv(find_events(vid),low_memory=False)
        ev=cache[vid]
        s=float(r.requested_start_sec); e=float(r.requested_end_sec)
        ss=pd.to_numeric(ev['start_sec'],errors='coerce'); ee=pd.to_numeric(ev['end_sec'],errors='coerce')
        near=ev[(ee>=s-4.0)&(ss<=e+4.0)].sort_values(['start_sec','end_sec'],kind='stable')
        cap=' '.join(near['text'].astype(str).tolist())
        ratio,proposal=best_subseq(str(r.text_pilot),cap)
        rows.append({
            'review_order':int(r.review_order),'verdict':str(r.verdict),'video_id':vid,'clip_id':str(r.clip_id),
            'whisper_text':str(r.text_pilot),'proposal_similarity':ratio,'proposed_caption_text':proposal
        })
    out=pd.DataFrame(rows).sort_values('review_order')
    OUT.mkdir(parents=True,exist_ok=True)
    out.to_csv(OUT/'PROPOSED_CANONICAL_TEXT.csv',index=False,encoding='utf-8-sig')
    stats={}
    for v,g in out.groupby('verdict'):
        x=pd.to_numeric(g['proposal_similarity'],errors='coerce').dropna()
        stats[str(v)]={'count':int(len(g)),'mean':float(x.mean()),'p10':float(x.quantile(.1)),'p50':float(x.quantile(.5)),'p90':float(x.quantile(.9))}
    summary={'stage':'SOURCE16_CAPTION_CANONICALIZATION_DIAGNOSTIC_V1','rows':int(len(out)),'by_verdict':stats}
    (OUT/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    print('\n===== BAD_TEXT PROPOSALS =====')
    bad=out[out['verdict'].isin(['BAD_TEXT','BAD_BOTH'])]
    print(bad[['review_order','video_id','clip_id','proposal_similarity','whisper_text','proposed_caption_text']].to_string(index=False))
    print('\nOUTPUT',OUT)
    return 0

if __name__=='__main__': raise SystemExit(main())
