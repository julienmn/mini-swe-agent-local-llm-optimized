import json
from unittest.mock import Mock, patch

import pytest
import requests

from minisweagent.exceptions import FormatError
from minisweagent.models import get_model_class
from minisweagent.models.ollama_model import OllamaAPIError, OllamaModel
from minisweagent.models.utils.actions_toolcall import BASH_TOOL


def _mock_response(payload: dict):
    response = Mock()
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    return response


def test_ollama_model_selection_shortcut():
    assert get_model_class("qwen3-coder:30b", "ollama") is OllamaModel


def test_ollama_model_requires_plain_model_name():
    with pytest.raises(ValueError, match="plain Ollama model name"):
        OllamaModel(model_name="ollama/qwen3-coder:30b")


def test_query_sends_native_chat_tools():
    payload = {
        "message": {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"function": {"name": "bash", "arguments": {"command": "echo test"}}}],
        },
        "prompt_eval_count": 10,
        "eval_count": 5,
    }
    model = OllamaModel(
        model_name="qwen3-coder:30b", base_url="http://localhost:11434", model_kwargs={"temperature": 0}
    )

    with patch("requests.post", return_value=_mock_response(payload)) as mock_post:
        result = model.query([{"role": "user", "content": "test"}])

    mock_post.assert_called_once()
    assert mock_post.call_args.args[0] == "http://localhost:11434/api/chat"
    assert mock_post.call_args.kwargs["timeout"] == 600
    request = json.loads(mock_post.call_args.kwargs["data"])
    assert request["model"] == "qwen3-coder:30b"
    assert request["messages"] == [{"role": "user", "content": "test"}]
    assert request["stream"] is False
    assert request["tools"] == [BASH_TOOL]
    assert request["options"] == {"temperature": 0}
    assert "Produce JSON OUTPUT ONLY" not in mock_post.call_args.kwargs["data"]
    assert result["extra"]["actions"] == [{"command": "echo test", "tool_call_id": "call_ollama_0"}]
    assert result["extra"]["cost"] == 0.0


def test_query_uses_config_timeout_without_sending_option():
    payload = {
        "message": {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"function": {"name": "bash", "arguments": {"command": "echo test"}}}],
        },
    }
    model = OllamaModel(model_name="qwen3-coder:30b", timeout=123, model_kwargs={"temperature": 0})

    with patch("requests.post", return_value=_mock_response(payload)) as mock_post:
        model.query([{"role": "user", "content": "test"}])

    request = json.loads(mock_post.call_args.kwargs["data"])
    assert mock_post.call_args.kwargs["timeout"] == 123
    assert request["options"] == {"temperature": 0}


def test_litellm_only_options_are_not_sent_to_ollama():
    payload = {
        "message": {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"function": {"name": "bash", "arguments": {"command": "echo test"}}}],
        },
    }
    model = OllamaModel(model_name="qwen3-coder:30b", model_kwargs={"drop_params": True, "temperature": 0})

    with patch("requests.post", return_value=_mock_response(payload)) as mock_post:
        model.query([{"role": "user", "content": "test"}])

    request = json.loads(mock_post.call_args.kwargs["data"])
    assert request["options"] == {"temperature": 0}


def test_timeout_from_env(monkeypatch):
    monkeypatch.setenv("MSWEA_OLLAMA_TIMEOUT", "321")

    assert OllamaModel(model_name="qwen3-coder:30b").config.timeout == 321
    assert OllamaModel(model_name="qwen3-coder:30b", timeout=123).config.timeout == 123


def test_query_text_sends_no_tools_and_maps_max_tokens():
    payload = {
        "message": {"role": "assistant", "content": "summary"},
        "prompt_eval_count": 4,
        "eval_count": 2,
    }
    model = OllamaModel(model_name="qwen3-coder:30b")

    with patch("requests.post", return_value=_mock_response(payload)) as mock_post:
        result = model.query_text([{"role": "user", "content": "summarize"}], max_tokens=100)

    request = json.loads(mock_post.call_args.kwargs["data"])
    assert "tools" not in request
    assert request["options"] == {"num_predict": 100}
    assert result["choices"][0]["message"]["content"] == "summary"
    assert result["usage"] == {"prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6}


def test_prepare_messages_converts_tool_history_for_ollama():
    model = OllamaModel(model_name="qwen3-coder:30b")
    messages = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "bash", "arguments": '{"command": "echo old"}'},
                }
            ],
            "extra": {"actions": []},
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
    ]

    prepared = model._prepare_messages_for_api(messages)

    assert "extra" not in prepared[0]
    assert prepared[0]["tool_calls"][0]["function"]["arguments"] == {"command": "echo old"}
    assert prepared[1] == {"role": "tool", "content": "ok", "tool_name": "bash"}


def test_missing_tool_calls_raises_format_error():
    model = OllamaModel(model_name="qwen3-coder:30b")
    payload = {"message": {"role": "assistant", "content": "no tool"}}

    with patch("requests.post", return_value=_mock_response(payload)):
        with pytest.raises(FormatError):
            model.query([{"role": "user", "content": "test"}])


def test_format_observation_messages_adds_tool_name():
    model = OllamaModel(model_name="qwen3-coder:30b", observation_template="{{ output.output }}")
    message = {"extra": {"actions": [{"command": "echo test", "tool_call_id": "call_1"}]}}
    result = model.format_observation_messages(message, [{"output": "test output", "returncode": 0}])

    assert result == [
        {
            "content": "test output",
            "extra": {
                "raw_output": "test output",
                "returncode": 0,
                "timestamp": result[0]["extra"]["timestamp"],
                "exception_info": None,
            },
            "tool_call_id": "call_1",
            "role": "tool",
            "tool_name": "bash",
        }
    ]


def test_base_url_from_env_and_config(monkeypatch):
    monkeypatch.setenv("OLLAMA_API_BASE", "http://env-host:11434")
    assert OllamaModel(model_name="qwen3-coder:30b")._api_url == "http://env-host:11434/api/chat"
    assert (
        OllamaModel(model_name="qwen3-coder:30b", base_url="http://config-host:11434")._api_url
        == "http://config-host:11434/api/chat"
    )


def test_http_errors_raise_ollama_error():
    response = Mock()
    response.status_code = 500
    response.text = "server error"
    response.raise_for_status.side_effect = requests.exceptions.HTTPError()
    model = OllamaModel(model_name="qwen3-coder:30b")

    with patch("requests.post", return_value=response):
        with pytest.raises(OllamaAPIError, match="HTTP 500: server error"):
            model._query([{"role": "user", "content": "test"}])
