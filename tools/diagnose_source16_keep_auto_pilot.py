from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / 'outputs' / 'source16_keep_auto_pilot_review' / 'REVIEW_RESULTS.csv'
PILOT = ROOT / 'kaggle_output' / 'source16-keep-auto-cut-pilot' / 'source16_keep_auto_cut_pilot' / 'outputs' / 'LISTENING_INDEX.csv'
WHISPER_ROOT = ROOT / 'kaggle_output' / 'source16-whisper-consensus'
OUT = ROOT / 'outputs' / 'source16_keep_auto_pilot_diagnostic'

def tokens(s: str) -> int:
    return len([x for x in str(s).strip().split() if x])

def find_source_dir(video_id: str) -> Path:
    matches = [p for p in WHISPER_ROOT.rglob(video_id) if p.is_dir()]
    for p in matches:
        if (p / 'WHISPER_WORDS.csv').exists() and (p / 'WHISPER_SEGMENTS.csv').exists():
            return p
    raise FileNotFoundError(f'Whisper source dir not found for {video_id} under {WHISPER_ROOT}')

def overlap_text(df: pd.DataFrame, start: float, end: float, text_col: str) -> str:
    s = pd.to_numeric(df['start_sec'], errors='coerce')
    e = pd.to_numeric(df['end_sec'], errors='coerce')
    sub = df[(e > start) & (s < end)].copy()
    if sub.empty:
        return ''
    sub = sub.sort_values(['start_sec','end_sec'], kind='stable')
    return ' '.join(sub[text_col].astype(str).tolist()).strip()

def main() -> int:
    if not REVIEW.exists():
        raise SystemExit(f'Missing review results: {REVIEW}')
    if not PILOT.exists():
        raise SystemExit(f'Missing pilot index: {PILOT}')

    review = pd.read_csv(REVIEW, low_memory=False).fillna('')
    pilot = pd.read_csv(PILOT, low_memory=False)
    reviewed = review[review['verdict'].astype(str).str.len() > 0].copy()
    if reviewed.empty:
        raise SystemExit('No reviewed rows yet.')

    pcols = ['clip_id','requested_start_sec','requested_end_sec','requested_duration_sec','actual_duration_sec','text']
    merged = reviewed.merge(pilot[pcols], on='clip_id', how='left', suffixes=('_review','_pilot'))
    rows = []
    cache = {}

    for r in merged.itertuples(index=False):
        video_id = str(r.video_id)
        if video_id not in cache:
            src = find_source_dir(video_id)
            words = pd.read_csv(src / 'WHISPER_WORDS.csv', low_memory=False)
            segs = pd.read_csv(src / 'WHISPER_SEGMENTS.csv', low_memory=False)
            cache[video_id] = (words, segs)
        words, segs = cache[video_id]

        start = float(r.requested_start_sec)
        end = float(r.requested_end_sec)
        duration = float(r.actual_duration_sec)
        word_text = overlap_text(words, start, end, 'word')
        seg_text = overlap_text(segs, start, end, 'text')
        pilot_text = str(r.text_pilot)
        wc = tokens(word_text)
        sc = tokens(seg_text)
        pc = tokens(pilot_text)

        rows.append({
            'review_order': int(r.review_order),
            'video_id': video_id,
            'clip_id': str(r.clip_id),
            'verdict': str(r.verdict),
            'duration_sec': duration,
            'pilot_text': pilot_text,
            'pilot_token_count': pc,
            'word_overlap_text': word_text,
            'word_token_count': wc,
            'segment_overlap_text': seg_text,
            'segment_token_count': sc,
            'word_tokens_per_sec': (wc / duration) if duration > 0 else None,
            'segment_to_word_token_ratio': (sc / wc) if wc > 0 else None,
            'word_vs_pilot_same': word_text.strip() == pilot_text.strip(),
            'flags': str(r.flags),
            'note': str(r.note),
        })

    out = pd.DataFrame(rows).sort_values('review_order')
    OUT.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT / 'REVIEWED_DIAGNOSTIC.csv', index=False, encoding='utf-8-sig')

    bad = out[out['verdict'].isin(['BAD_TEXT','BAD_BOTH'])].copy()
    suspicious = out[(out['word_tokens_per_sec'] < 0.8) | (out['segment_to_word_token_ratio'] > 1.8)].copy()
    bad.to_csv(OUT / 'BAD_TEXT_DIAGNOSTIC.csv', index=False, encoding='utf-8-sig')
    suspicious.to_csv(OUT / 'SUSPICIOUS_TIMESTAMP_TEXT.csv', index=False, encoding='utf-8-sig')

    summary = {
        'stage': 'SOURCE16_KEEP_AUTO_PILOT_DIAGNOSTIC_V1',
        'reviewed_rows': int(len(out)),
        'verdict_counts': {str(k): int(v) for k,v in out['verdict'].value_counts().to_dict().items()},
        'bad_text_or_both': int(len(bad)),
        'suspicious_timestamp_text_rows': int(len(suspicious)),
        'heuristics': {
            'low_word_density_tokens_per_sec_lt': 0.8,
            'segment_to_word_token_ratio_gt': 1.8,
        },
    }
    (OUT / 'SUMMARY.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print('\n===== BAD TEXT / BOTH =====')
    show = bad[['review_order','video_id','clip_id','duration_sec','pilot_token_count','word_token_count','segment_token_count','word_tokens_per_sec','segment_to_word_token_ratio','pilot_text','segment_overlap_text']]
    if len(show):
        print(show.to_string(index=False))
    else:
        print('(none)')
    print('\nOUTPUT', OUT)
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
