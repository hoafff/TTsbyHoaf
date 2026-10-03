# Kaggle jobs

This directory will contain reproducible Kaggle kernel/script jobs.

Planned split:

- `cpu/`: source/transcript audit and other non-GPU stages.
- `gpu/`: Whisper/alignment and later Piper training.
- `tools/`: submit/status/log/output wrappers.

The local control environment should submit jobs through Kaggle CLI rather than relying on an interactive browser notebook.
