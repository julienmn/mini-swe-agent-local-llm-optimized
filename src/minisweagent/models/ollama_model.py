import json
import logging
import os
import time
from typing import Any, Literal

import requests
from pydantic import BaseModel, Field

from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.utils.actions_toolcall import (
    BASH_TOOL,
    format_toolcall_observation_messages,
    parse_toolcall_actions,
)
from minisweagent.models.utils.openai_multimodal import expand_multimodal_content

logger = logging.getLogger("ollama_model")

LITELLM_ONLY_OPTIONS = {"drop_params"}
ThinkingLevel = Literal["low", "medium", "high", "max"]
ThinkingMode = bool | ThinkingLevel


class OllamaModelConfig(BaseModel):
    model_name: str
    """Plain Ollama model name, e.g. qwen3-coder:30b."""
    base_url: str = Field(default_factory=lambda: os.getenv("OLLAMA_API_BASE", "http://localhost:11434"))
    """Ollama server base URL."""
    timeout: float = Field(default_factory=lambda: float(os.getenv("MSWEA_OLLAMA_TIMEOUT", "600")))
    """HTTP request timeout in seconds."""
    think: ThinkingMode | None = Field(
        default_factory=lambda: os.getenv("MSWEA_OLLAMA_THINK"), validate_default=True
    )
    """Ollama thinking mode. None leaves the model default unchanged."""
    compaction_think: ThinkingMode = Field(
        default_factory=lambda: os.getenv("MSWEA_OLLAMA_COMPACTION_THINK", "false"), validate_default=True
    )
    """Thinking mode for internal context-compaction summaries."""
    stream: bool = Field(default_factory=lambda: os.getenv("MSWEA_OLLAMA_STREAM", "true"), validate_default=True)
    """Stream normal model responses from Ollama."""
    compaction_stream: bool = False
    """Stream internal context-compaction summaries."""
    model_kwargs: dict[str, Any] = {}
    """Runtime options passed to Ollama's options object."""
    format_error_template: str = "{{ error }}"
    """Template used when the LM's output is not in the expected format."""
    observation_template: str = (
        "{% if output.exception_info %}<exception>{{output.exception_info}}</exception>\n{% endif %}"
        "<returncode>{{output.returncode}}</returncode>\n<output>\n{{output.output}}</output>"
    )
    """Template used to render the observation after executing an action."""
    multimodal_regex: str = ""
    """Regex to extract multimodal content. Empty string disables multimodal processing."""

    def model_post_init(self, __context: Any) -> None:
        if self.model_name.startswith("ollama/") or self.model_name.startswith("ollama_chat/"):
            raise ValueError("OllamaModel expects a plain Ollama model name, e.g. qwen3-coder:30b.")


class OllamaAPIError(Exception):
    """Custom exception for Ollama API errors."""


class OllamaModel:
    abort_exceptions: list[type[Exception]] = [OllamaAPIError, KeyboardInterrupt]

    def __init__(self, **kwargs):
        self.config = OllamaModelConfig(**kwargs)
        self._api_url = self.config.base_url.rstrip("/") + "/api/chat"
        self._last_provider_request = None
        self._last_provider_response = None

    def _options(self, kwargs: dict) -> dict:
        options = self.config.model_kwargs | kwargs
        for key in LITELLM_ONLY_OPTIONS | {"think"}:
            options.pop(key, None)
        max_tokens = options.pop("max_tokens", None)
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        max_completion_tokens = options.pop("max_completion_tokens", None)
        if max_completion_tokens is not None:
            options["num_predict"] = max_completion_tokens
        return options

    def _query(
        self, messages: list[dict[str, str]], *, tools: bool = True, readable_debug_callback=None, **kwargs
    ) -> dict:
        think = kwargs.pop("think", self.config.think)
        stream = kwargs.pop("stream", self.config.stream)
        stream_callback = kwargs.pop("stream_callback", None)
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "stream": stream,
            "options": self._options(kwargs),
        }
        if tools:
            payload["tools"] = [BASH_TOOL]
        if think is not None:
            payload["think"] = think
        body = json.dumps(payload)
        headers = {"Content-Type": "application/json"}
        self._last_provider_request = {
            "provider": "ollama",
            "method": "POST",
            "url": self._api_url,
            "headers": headers,
            "body": body,
            "payload": payload,
            "timeout": self.config.timeout,
        }
        self._last_provider_response = None

        try:
            if readable_debug_callback:
                readable_debug_callback("request", provider="ollama", payload=payload)
            response = requests.post(
                self._api_url,
                headers=headers,
                data=body,
                timeout=self.config.timeout,
                stream=stream,
            )
            response.raise_for_status()
            response_json = self._collect_stream(response, stream_callback) if stream else response.json()
            self._last_provider_response = response_json
            if readable_debug_callback:
                readable_debug_callback("response", provider="ollama", response=response_json)
            return response_json
        except requests.exceptions.HTTPError as e:
            self._last_provider_response = {
                "status_code": response.status_code,
                "text": response.text,
            }
            if readable_debug_callback:
                readable_debug_callback("error", provider="ollama", error=repr(e))
            raise OllamaAPIError(f"HTTP {response.status_code}: {response.text}") from e
        except requests.exceptions.RequestException as e:
            self._last_provider_response = {"error": repr(e)}
            if readable_debug_callback:
                readable_debug_callback("error", provider="ollama", error=repr(e))
            raise OllamaAPIError(f"Request failed: {e}") from e

    @staticmethod
    def _collect_stream(response, stream_callback) -> dict:
        result, message = {}, {}
        for line in response.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            delta = chunk.get("message", {})
            for key, value in chunk.items():
                if key != "message":
                    result[key] = value
            for key in ("content", "thinking"):
                if key in delta:
                    message[key] = message.get(key, "") + (delta[key] or "")
            if delta.get("tool_calls"):
                message.setdefault("tool_calls", []).extend(delta["tool_calls"])
            for key, value in delta.items():
                if key not in {"content", "thinking", "tool_calls"}:
                    message[key] = value
            if stream_callback and delta:
                stream_callback(delta)
        result["message"] = message
        return result

    def query_text(self, messages: list[dict[str, str]], **kwargs) -> dict:
        response = self._query(self._prepare_messages_for_api(messages), tools=False, **kwargs)
        message = response.get("message", {})
        return {"choices": [{"message": message}], "usage": self._usage(response)}

    def compaction_query_kwargs(self) -> dict:
        return {"think": self.config.compaction_think, "stream": self.config.compaction_stream}

    @staticmethod
    def stream_query_kwargs(callback) -> dict:
        return {"stream_callback": callback}

    def _prepare_messages_for_api(self, messages: list[dict]) -> list[dict]:
        prepared = []
        for message in messages:
            msg = {k: v for k, v in message.items() if k != "extra"}
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                msg["tool_calls"] = [self._tool_call_for_ollama(tool_call) for tool_call in msg["tool_calls"]]
            if msg.get("role") == "tool":
                msg.pop("tool_call_id", None)
                msg.setdefault("tool_name", "bash")
            prepared.append(msg)
        return prepared

    def _tool_call_for_ollama(self, tool_call: dict) -> dict:
        function = dict(tool_call.get("function", {}))
        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"command": arguments}
        function["arguments"] = arguments
        return {"type": tool_call.get("type", "function"), "function": function}

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        response = self._query(self._prepare_messages_for_api(messages), **kwargs)
        cost_output = self._calculate_cost(response)
        GLOBAL_MODEL_STATS.add(cost_output["cost"])
        message = dict(response.get("message", {}))
        message["extra"] = {
            "actions": self._parse_actions(response),
            "response": response,
            **cost_output,
            "timestamp": time.time(),
        }
        return message

    def _usage(self, response: dict) -> dict:
        prompt_tokens = response.get("prompt_eval_count", 0)
        completion_tokens = response.get("eval_count", 0)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    def _calculate_cost(self, response) -> dict[str, float]:
        return {"cost": 0.0}

    def _parse_actions(self, response: dict) -> list[dict]:
        tool_calls = response.get("message", {}).get("tool_calls") or []
        tool_calls = [_DictToObj(tc, i) for i, tc in enumerate(tool_calls)]
        return parse_toolcall_actions(tool_calls, format_error_template=self.config.format_error_template)

    def format_message(self, **kwargs) -> dict:
        return expand_multimodal_content(kwargs, pattern=self.config.multimodal_regex)

    def format_observation_messages(
        self, message: dict, outputs: list[dict], template_vars: dict | None = None
    ) -> list[dict]:
        actions = message.get("extra", {}).get("actions", [])
        messages = format_toolcall_observation_messages(
            actions=actions,
            outputs=outputs,
            observation_template=self.config.observation_template,
            template_vars=template_vars,
            multimodal_regex=self.config.multimodal_regex,
        )
        for msg in messages:
            if msg.get("role") == "tool":
                msg["tool_name"] = "bash"
        return messages

    def get_template_vars(self, **kwargs) -> dict[str, Any]:
        return self.config.model_dump()

    def serialize(self) -> dict:
        return {
            "info": {
                "config": {
                    "model": self.config.model_dump(mode="json"),
                    "model_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
            }
        }


class _DictToObj:
    """Simple wrapper to convert Ollama tool call dicts to the parser's expected shape."""

    def __init__(self, d: dict, index: int = 0):
        self._d = d
        self.id = d.get("id") or f"call_ollama_{index}"
        self.function = _DictToObj(d.get("function", {})) if "function" in d else None
        self.name = d.get("name")
        arguments = d.get("arguments")
        self.arguments = json.dumps(arguments) if isinstance(arguments, dict) else arguments
