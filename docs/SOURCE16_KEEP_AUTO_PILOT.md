# Source16 KEEP_AUTO 22050 Hz Pilot

This is the gate for bulk-cutting the 17,495 KEEP_AUTO segments.

The earlier 160-row mixed pilot intentionally contained KEEP_AUTO, KEEP_WHISPER_PRIMARY, and REVIEW rows and preserved source sample rate. It remains useful only as a diagnostic artifact; it must not be used to approve the KEEP_AUTO bulk dataset.

The corrected pilot:

- selects exactly 160 rows;
- selects only `KEEP_AUTO`;
- balances the sample across the 11 sources that actually contain KEEP_AUTO rows;
- includes quality-spread, shortest, longest, and weakest-boundary examples;
- cuts to WAV / PCM16 / mono / 22,050 Hz;
- verifies every output with ffprobe;
- fails if any output is not 22,050 Hz mono or if duration is materially wrong.

Run locally:

```powershell
python .\tools\stage_source16_keep_auto_pilot.py
python .\tools\kaggle_bridge.py submit source16-keep-auto-cut-pilot
python .\tools\kaggle_bridge.py watch source16-keep-auto-cut-pilot
python .\tools\kaggle_bridge.py output source16-keep-auto-cut-pilot
python .\tools\review_source16_keep_auto_pilot.py
```

The listening review is checkpointed under:

`outputs/source16_keep_auto_pilot_review/`

Only after this KEEP_AUTO-only 22,050 Hz pilot passes listening review should the 17,495 KEEP_AUTO rows be bulk-cut for training.
