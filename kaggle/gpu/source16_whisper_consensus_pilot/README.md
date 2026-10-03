# source16-whisper-consensus-pilot

GPU pilot for one locked source: `Pf9QTip2hqI`.

This job:
- downloads the source audio from YouTube;
- downloads Vietnamese YouTube automatic captions as raw `json3`;
- runs Faster-Whisper `large-v3` in Vietnamese with word timestamps and VAD;
- exports Whisper segment and word timing tables;
- aligns each Whisper segment with overlapping YouTube caption events;
- computes diagnostic lexical similarity only;
- exports pause/word-duration features for later automatic prosody QC.

It does **not** modify the existing `semantic_v2` queue or results, and it does not use GPT.

Run:

```powershell
python .\tools\kaggle_bridge.py submit source16-whisper-consensus-pilot
python .\tools\kaggle_bridge.py logs source16-whisper-consensus-pilot -f
python .\tools\kaggle_bridge.py output source16-whisper-consensus-pilot
```

Key outputs:
- `YOUTUBE_CAPTION_EVENTS.csv`
- `WHISPER_SEGMENTS.csv`
- `WHISPER_WORDS.csv`
- `PAUSE_FEATURES.csv`
- `CONSENSUS_WINDOWS.csv`
- `SUMMARY.json`
- raw YouTube `*.vi.json3`
