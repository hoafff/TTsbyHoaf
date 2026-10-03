# LLM semantic pilot executor

Runs the semantic pilot through any provider exposing an OpenAI-compatible chat-completions endpoint.

Required environment variables:
- LLM_BASE_URL
- LLM_API_KEY
- LLM_MODEL

The executor is resumable. It saves each pilot item separately and validates exact source-text coverage before accepting a result.

Start with 24 items, inspect quality, then expand.
