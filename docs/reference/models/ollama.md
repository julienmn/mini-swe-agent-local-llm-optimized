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
```

The model name must be the plain Ollama model name, not a LiteLLM provider string such as `ollama/qwen3-coder:30b`.
`MSWEA_OLLAMA_TIMEOUT` configures the native Ollama HTTP request timeout; `LITELLM_TIMEOUT` does not apply to this model class.

:::: minisweagent.models.ollama_model

--8<-- "docs/_footer.md"
