# Architecture

## Design goal

Tách control plane khỏi compute plane để không phải giữ Kaggle Draft Session mở và không đốt accelerator quota cho các bước CPU.

## Control plane

Local VS Code + GitHub repo:
- code
- config
- pipeline state
- submit/status/log/download helpers

## Compute plane

Kaggle jobs:
- CPU jobs: source audit, transcript pack, text cleaning prep, manifests
- GPU jobs: ASR/alignment and later Piper training
- each job should checkpoint and exit when its stage is done

## Durable storage

### GitHub
Only small text/code artifacts.

### Hugging Face private
- Piper checkpoints
- QC/manifests
- transcript review packs
- intermediate stage backups

### Kaggle Dataset
- compressed raw source audio
- final train dataset version(s)

## Source-to-training flow

1. Lock 16 source IDs.
2. Establish canonical timestamped text per source.
3. Split review packs near 50k characters, always on sentence boundaries.
4. AI/text QC.
5. Semantic sentence planning.
6. Fresh align/cut from original source.
7. Boundary/VAD/audio-quality checks.
8. ASR verification.
9. KEEP-only train metadata.
10. Mini validation train from epoch 9 before scaling.

## Important text rule

The character budget is a packaging target, never a sentence splitter. A review part may exceed the target to preserve a complete sentence.
