# AGENTS.md — HARD PROJECT RULES

## READ THIS FIRST

These rules are mandatory for every coding or automation session in this repository.
Do not start a long, expensive, destructive, or hard-to-reproduce run before checking this file.

## 1. Durable checkpoint is mandatory

A durable checkpoint MUST exist for any run that is:

- expected to take 2 hours or more;
- uses limited/paid GPU or other scarce compute;
- processes many independent sources/items;
- produces expensive-to-recompute results;
- changes or consumes important project data;
- otherwise marked as a critical run.

A directory inside an ephemeral worker such as `/kaggle/working` is NOT a checkpoint.

The checkpoint must live outside the worker/session, for example in a persistent Kaggle Dataset,
Drive/object storage, Git/Git LFS when appropriate, or another durable store.

## 2. What a checkpoint must contain

At minimum:

- completed unit/source IDs;
- enough expensive intermediate artifacts to continue without recomputing completed work;
- configuration fingerprint/version (model, beam size, source set, important processing parameters);
- per-unit summary/status;
- failures/errors that matter for resume;
- schema/version for the checkpoint format.

Do NOT checkpoint bulky inputs that already exist durably elsewhere unless they are required for recovery.

## 3. Resume must be proven before the full run

Before spending substantial GPU/compute time:

1. restore or initialize the durable checkpoint;
2. perform a real write to the durable checkpoint backend;
3. confirm the write succeeded;
4. only then load/start expensive compute;
5. for a new pipeline, test restart/resume on a small pilot before the full run.

If checkpoint restore/write fails, STOP. Do not continue the expensive run.

## 4. Save after every recoverable unit

For independent sources/items, persist the result immediately after that unit finishes.
Do not start additional queued work after a unit completes until that completed unit is durably checkpointed.

With parallel workers, a crash may lose only the currently in-flight units. Already checkpointed units must never be recomputed.

## 5. Never mix incompatible checkpoints

Resume only when checkpoint configuration matches the current run.
If model, decoding parameters, source set, schema, or other result-affecting settings changed,
fail closed and create/migrate a clearly versioned checkpoint instead of silently mixing results.

## 6. Long-run preflight is mandatory

Before submit/start:

- validate/compile the code;
- verify required inputs exist;
- verify durable checkpoint read + write;
- verify resume plan;
- ensure no conflicting run is already active;
- print the effective important parameters to logs.

Never rely on hard-coded summary metadata that disagrees with runtime parameters.

## 7. Preserve expensive results; recompute cheap derivatives

Checkpoint the smallest set of artifacts that avoids repeating expensive work.
Cheap deterministic derivatives should be rebuilt later from the checkpointed expensive outputs.

For Whisper pipelines this normally means preserving transcription/timestamp outputs and metadata,
while CPU-only tables, slicing manifests, and other deterministic derivatives may be regenerated.

## 8. Stop/cancel discipline

Stopping a local watcher is not the same as stopping a remote job.
Before submitting a replacement run, explicitly check the remote job status and cancel/stop the old run if needed.

## 9. No exceptions by convenience

Do not skip these rules because a run "should finish soon", "probably will not disconnect", or has succeeded before.
If a run qualifies as long/critical, checkpoint + resume safety is part of the implementation, not an optional improvement.

## 10. TTS project decision ledger

For Source16/TTS data-preparation work, read `docs/TTS_PIPELINE_DECISIONS.md` before proposing, implementing, or changing pipeline behavior. Treat that file as the durable project-memory ledger for user-approved decisions. Update it whenever a later user decision changes those rules.
