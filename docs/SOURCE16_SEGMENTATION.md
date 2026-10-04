# Source16 CPU Segmentation Planner

This stage turns the durable Source16 Whisper outputs into a proposed TTS cut manifest.

It does **not** cut audio.

## Inputs

- ahndongo/maymay-source16-media
- ahndongo/maymay-source16-whisper-checkpoint

The job fails closed unless the Whisper checkpoint is exactly 16/16 and matches:

- stage SOURCE16_WHISPER_CONSENSUS_V1
- model large-v3
- beam size 1
- the locked 16-source list

## Boundary policy

Candidate boundaries come only from Whisper word timestamps.

The planner prefers:

- terminal punctuation;
- pauses of about 0.20 s or more;
- Whisper segment boundaries;
- durations near the 8 s target.

Default duration envelope:

- minimum: 3 s
- target: 8 s
- normal maximum: 14 s
- absolute maximum: 16 s

Small padding is proposed around word boundaries, capped by available silence so adjacent clips do not blindly overlap.

## Quality policy

The manifest records:

- mean and p10 Whisper word probability;
- median segment avg_logprob;
- maximum segment no_speech_prob;
- rebuilt 30-second YouTube-caption/Whisper consensus;
- source-level caption trust;
- per-candidate decision and reason.

YouTube captions are only a quality signal. If a source has poor global caption agreement, the planner switches to KEEP_WHISPER_PRIMARY instead of automatically rejecting good Whisper candidates.

Decisions:

- KEEP_AUTO
- KEEP_WHISPER_PRIMARY
- REVIEW
- DROP

## Outputs

- outputs/CUT_MANIFEST.csv — all candidates
- outputs/CUT_MANIFEST_KEEP.csv — keep candidates only
- outputs/REVIEW_QUEUE.csv — review candidates, lowest quality first
- outputs/SOURCE_SUMMARY.csv
- outputs/SUMMARY.json
- outputs/sources/<video_id>/CANDIDATES.csv
- outputs/sources/<video_id>/CONSENSUS_BUCKETS_30S.csv
- outputs/sources/<video_id>/SUMMARY.json

Review these outputs before any bulk ffmpeg slicing.

This stage is CPU-only, deterministic, and cheap to regenerate from the durable Whisper checkpoint and durable media dataset, so it intentionally does not create a second checkpoint dataset.
