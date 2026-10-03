# Semantic task builder V2

V1 revealed that many long caption blocks already contain multiple internal sentence-ending punctuation marks.

V2 therefore performs a deterministic, lossless split on internal terminal punctuation before any LLM call.

Key properties:
- exact source text is preserved;
- each piece stores exact character offsets inside its parent caption block;
- each AI task contains one source video only;
- proportional timestamps are only rough hints;
- final audio boundaries will be determined later by alignment against raw audio;
- the LLM may only split further and label, not rewrite text.

This reduces token cost and makes source-text provenance easier to validate.
