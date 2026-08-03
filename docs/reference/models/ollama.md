# OllamaModel

Direct integration with Ollama's `/api/chat` endpoint for local models.

Use this model class when you want mini-SWE-agent to talk to Ollama directly instead of routing through LiteLLM:

```bash
MSWEA_MODEL_CLASS=ollama
MSWEA_MODEL_NAME=qwen3-coder:30b
OLLAMA_API_BASE=http://localhost:11434
MSWEA_OLLAMA_TIMEOUT=600
```

Or in an agent config file:

```yaml
model:
  model_class: ollama
  model_name: qwen3-coder:30b
  base_url: http://localhost:11434
  timeout: 600
  think: true
  compaction_think: false
  stream: true
  compaction_stream: false
```

The model name must be the plain Ollama model name, not a LiteLLM provider string such as `ollama/qwen3-coder:30b`.
`MSWEA_OLLAMA_TIMEOUT` configures the native Ollama HTTP request timeout; `LITELLM_TIMEOUT` does not apply to this model class.

`think` is passed to Ollama's top-level chat request field. Set it to `true`, `false`, or a model-supported level such as `low`, `medium`, `high`, or `max`; omit it to keep the model default. `compaction_think` controls internal context-summary calls and defaults to `false` so summaries do not spend their token budget on reasoning.

`stream` displays normal model responses as Ollama produces them. It defaults to `true`; `compaction_stream` defaults to `false`.

The same settings can be configured globally with `MSWEA_OLLAMA_THINK`, `MSWEA_OLLAMA_COMPACTION_THINK`, and `MSWEA_OLLAMA_STREAM`.

:::: minisweagent.models.ollama_model

--8<-- "docs/_footer.md"
