# source16-whisper-consensus

GPU stage for the locked 16 MayMay sources.

Inputs:
- private Kaggle dataset `ahndongo/maymay-runtime-secrets`
- required: `cookies.txt`
- optional: `hf_token.txt`

For each source this job:
- downloads audio with yt-dlp using the private cookie file;
- downloads Vietnamese YouTube automatic captions as raw JSON3;
- runs Faster-Whisper `large-v3` once with word timestamps and VAD;
- exports Whisper segment/word tables;
- computes YouTube ↔ Whisper diagnostic lexical similarity;
- exports pause/word-duration features for later automatic prosody QC;
- deletes downloaded audio after each source to keep output small.

It does not use GPT and does not modify semantic_v2 review results.

Run only after the one-source pilot has completed successfully:

```powershell
python .\tools\kaggle_bridge.py submit source16-whisper-consensus
python .\tools\kaggle_bridge.py logs source16-whisper-consensus -f
python .\tools\kaggle_bridge.py output source16-whisper-consensus
```
