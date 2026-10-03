# GitHub review queue

This is the mainline review mode for MayMay semantic text QC.

There is **no external paid LLM API requirement**.

Workflow:

1. Local script exports compact, sentence-safe review packs from `PRESEGMENTED_PIECES.csv`.
2. User commits/pushes `review_queue/semantic_v2/` to GitHub.
3. ChatGPT reads review packs directly through the GitHub connector.
4. ChatGPT writes structured review results back into the repository.
5. Local validators merge/validate the committed results before timestamp mapping or audio work.

Each pack:
- contains one source video only;
- targets roughly 45k source-text characters;
- never splits an existing semantic piece just to hit the size target;
- includes stable piece IDs and small previous/next context;
- keeps AI review lossless: split/label only, no text rewriting yet.

The optional external API executor remains experimental/optional and is not the mainline.
