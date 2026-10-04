# TTS Pipeline Decision Ledger

This file records project decisions that must survive chat/session changes. Read it before changing Source16/TTS data-preparation logic.

## Current training-audio target

- Final training audio format: WAV, PCM signed 16-bit, mono, **22,050 Hz**.
- Any pilot intended to approve data for training must be evaluated in this final format, not at the source sample rate.

## Source16 pipeline stages

1. Preserve durable source media and captions/reference text.
2. Run expensive Whisper transcription/timestamps with durable checkpointing.
3. Build segmentation candidates from word timestamps, pauses, punctuation, and confidence.
4. Classify candidates:
   - `KEEP_AUTO`: strongest automatically accepted class.
   - `KEEP_WHISPER_PRIMARY`: Whisper is trusted while caption/reference text is not trusted enough to arbitrate.
   - `REVIEW`: candidate needs human/manual inspection.
   - `DROP`: reject.
5. Run structural QC.
6. Run a listening pilot for the class being considered for bulk cutting.
7. Only after the pilot passes, bulk-cut that class.
8. Run post-segmentation sanitation/filtering before final training-set export.

Do not collapse these stages into one another.

## What the current KEEP_AUTO pilot is testing

The corrected KEEP_AUTO pilot is **160 rows, KEEP_AUTO-only**, balanced across sources that contain KEEP_AUTO, and cut to 22,050 Hz mono PCM16.

Its primary purpose is to validate segmentation/cut quality:

- no clipped initial phoneme;
- no clipped final phoneme;
- no cut through the middle of a phrase/sentence;
- no unwanted fragment from the previous/next utterance;
- reasonable leading/trailing silence;
- punctuation/pause boundaries sound natural.

Text or source defects noticed during listening may be recorded, but they are not the primary purpose of this pilot.

## Text/reference interpretation

`KEEP_AUTO` does not mean every transcript has been manually proven word-for-word. It means the candidate passed the strongest automatic acceptance rules.

Do not describe a mixed pilot as “all KEEP” merely because it contains no DROP rows. State exact decision counts.

`KEEP_WHISPER_PRIMARY` is not automatically lower audio quality. It means source caption/reference alignment is too unreliable to use as the main arbiter, so Whisper/timestamps/confidence are primary.

## Post-segmentation sanitation

Fragment cleanup is a separate step after segmentation validation and before final training export.

Examples to detect/filter or recut:

- standalone one-word fragments;
- trailing/leading interjections after a terminal punctuation mark;
- one- or two-token fragments that do not form a useful training utterance;
- duplicated neighboring text;
- accidental partial words at boundaries.

Do not mark a real spoken word as `BAD_TEXT` merely because it is short. If audio and transcript both contain it, it is a content/fragment decision.

### Main-corpus policy for isolated one-word interjections

For the clean narrator/base TTS corpus, standalone one-word interjections such as “Hô.” are **excluded from the main training set by default** unless there is a deliberate phoneme/prosody-coverage reason to keep them.

If a one-word interjection is attached to an otherwise good segment and has an independent timestamp boundary, prefer recutting the segment to exclude that interjection from the main corpus rather than keeping it merely for “more data”.

Such material may be preserved in an auxiliary/expressive bucket for later experiments; do not destroy source evidence.

## Preserve evidence

Never rewrite or discard source evidence just to make the dataset look clean. Keep original timestamps, transcript evidence, decision reason, and source provenance so later filtering/re-cutting remains reproducible.

## Chat/session continuity rule

When continuing TTS/Source16 work after a context switch, inspect this ledger and `AGENTS.md` before proposing or changing pipeline behavior. If a new user decision changes one of these rules, update this ledger in the same logical phase/commit.
