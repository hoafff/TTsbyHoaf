# Caption structure QC

The first caption fetch produced complete 16-source data, but YouTube automatic punctuation is too sparse to use punctuation-only sentence boundaries as final TTS cuts.

The local QC tool measures:
- punctuation-unit character/word/duration distributions;
- events per punctuation unit;
- raw caption-event gap distributions;
- counts of candidate pause boundaries;
- the 200 longest punctuation units.

Run after downloading `source16-captions`:

```powershell
python .\tools\caption_qc.py
```

The result under `outputs/caption_qc_v1` is a calibration artifact for the semantic sentence planner, not a training manifest.
