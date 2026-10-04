# Source16 Cut Pilot

This stage performs the first real audio slicing step, but only for the 160-row listening pilot selected by local segmentation QC.

## Safety gates

Before upload, `tools/stage_source16_cut_pilot_manifest.py` requires:

- `QC_SUMMARY.json` with `qc_pass=true`;
- exactly 160 pilot rows;
- exactly 16 sources x 10 clips/source;
- no duplicate clip IDs;
- no DROP rows;
- positive, valid cut intervals.

The helper uploads a tiny private dataset:

`ahndongo/maymay-source16-cut-pilot-manifest-v1`

The Kaggle CPU job then mounts:

- `ahndongo/maymay-source16-media`
- `ahndongo/maymay-source16-cut-pilot-manifest-v1`

## Output

The cutter creates:

- `clips/<video_id>/<clip_id>.wav`
- `outputs/LISTENING_INDEX.csv`
- `outputs/FAILURES.csv`
- `outputs/SUMMARY.json`
- `outputs/README.txt`

WAV files are mono PCM16. The input sample rate is preserved for this listening pilot so the boundary test does not add a separate resampling decision.

Every clip is duration-checked with ffprobe. The job fails closed if any of the 160 clips cannot be cut or the measured duration differs materially from the requested interval.

## What to listen for

Before bulk cutting, inspect the pilot for:

- clipped initial phonemes;
- clipped final phonemes;
- cuts in the middle of a phrase/sentence;
- excessive leading/trailing silence;
- transcript/audio mismatch;
- behavior of REVIEW rows;
- behavior of KEEP_WHISPER_PRIMARY sources.

Only after listening QC passes should a bulk cutter be enabled.
