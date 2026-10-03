# source16-captions

CPU-only stage after `source16-audit`.

This job:
- downloads **Vietnamese automatic captions only** for the locked 16 sources;
- stores raw YouTube `json3` subtitle data with timestamps;
- parses every caption event into CSV;
- builds punctuation-based sentence units without modifying punctuation;
- creates AI-review text packs targeted at ~50k characters;
- **never splits a sentence unit merely to hit the size target**;
- adds 3 prior sentences as `CONTEXT_ONLY` between review parts.

It does **not** download media, run Whisper, cut WAVs, use V1 training clips, or train Piper.

Run:

```powershell
python .\tools\kaggle_bridge.py submit source16-captions
python .\tools\kaggle_bridge.py logs source16-captions -f
python .\tools\kaggle_bridge.py output source16-captions
```

Important: punctuation from YouTube auto captions is evidence, not canonical truth. The raw JSON3 data is retained so later cleaning/alignment can always trace back to source timestamps.
