# Durable Checkpoint & Long-Run Safety Policy

This document expands the mandatory rules in the repository-root `AGENTS.md`.

## Decision gate

A run is LONG/CRITICAL if any of the following is true:

- planned timeout or expected duration >= 2 hours;
- limited GPU/accelerator quota is used;
- many independent units are processed;
- recomputation cost is significant;
- the run mutates/consumes important data;
- the job registry explicitly sets `critical_run: true`.

No LONG/CRITICAL run may start without a durable, tested resume path.

## Required checkpoint properties

A valid checkpoint is:

- **durable**: survives worker/session/kernel termination;
- **incremental**: advances after each recoverable unit;
- **self-describing**: contains schema and result-affecting configuration;
- **validated**: completed units have the required artifacts;
- **compatible**: incompatible config causes a hard failure;
- **minimal**: stores expensive-to-recompute artifacts, not duplicate bulky inputs.

## Required lifecycle

```text
restore/init checkpoint
        |
real remote write preflight
        |
validate inputs + resume plan
        |
start expensive compute
        |
finish one unit
        |
persist that unit + state remotely
        |
only then queue replacement work
```

## Whisper-specific rule

For transcription pipelines, the durable checkpoint should preserve the GPU-expensive result:

- word/segment timestamps and text;
- transcription metadata;
- completed source IDs and config.

YouTube captions and source media already stored in a durable input dataset do not need to be duplicated.
CPU-derived pause/consensus tables and audio slicing manifests can be regenerated from the preserved timestamps.

## Restart acceptance test

A new long pipeline is not considered ready until a small pilot can:

1. complete at least one unit;
2. persist it remotely;
3. terminate/restart;
4. restore that unit;
5. skip it;
6. continue with the next unit.

If any step fails, the full run must not be launched.
