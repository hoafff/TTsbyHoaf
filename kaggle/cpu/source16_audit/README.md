# source16-audit

CPU-only Kaggle script job for the first mainline stage of the 16-source V2 rebuild.

It does **not** download source media, run ASR, use V1 clips, or train Piper.

It audits the locked 16 YouTube source IDs for:
- title/channel/duration;
- Vietnamese manual subtitles;
- Vietnamese automatic captions;
- whether the source should enter MANUAL_VI, AUTO_VI, or NEED_ASR.

Run from repo root:

```powershell
python .\tools\kaggle_bridge.py submit source16-audit
python .\tools\kaggle_bridge.py status source16-audit
python .\tools\kaggle_bridge.py logs source16-audit -f
python .\tools\kaggle_bridge.py output source16-audit
```
