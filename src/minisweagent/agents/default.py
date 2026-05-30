"""Basic agent class. See https://mini-swe-agent.com/latest/advanced/control_flow/ for visual explanation
or https://minimal-agent.com for a tutorial on the basic building principles.
"""

import json
import logging
import os
import re
import shlex
import time
import traceback
from pathlib import Path

from jinja2 import StrictUndefined, Template
from pydantic import BaseModel

from minisweagent import Environment, Model, __version__
from minisweagent.exceptions import FormatError, InterruptAgentFlow, LimitsExceeded, TimeExceeded
from minisweagent.models.utils.content_string import get_content_string
from minisweagent.utils.serialize import recursive_merge

compaction_logger = logging.getLogger("minisweagent.agent.compaction")

WHOLE_FILE_CAT_FORBIDDEN_OUTPUT = "Output of whole files IS FORBIDDEN. Use symbol based targeted reads instead."


def _shell_tokens(command: str) -> list[str]:
    lexer = shlex.shlex(command, posix=True, punctuation_chars="|&;<>")
    lexer.whitespace_split = True
    lexer.commenters = ""
    return list(lexer)


def _is_bounded_cat_pipe(tokens: list[str], pipe_index: int) -> bool:
    if pipe_index + 1 >= len(tokens):
        return False
    if tokens[pipe_index + 1] == "head":
        return True
    if tokens[pipe_index + 1] != "sed":
        return False
    sed_args = tokens[pipe_index + 2 :]
    if sed_args and sed_args[0] == "-n":
        sed_args = sed_args[1:]
    return bool(sed_args) and bool(re.fullmatch(r"\d+(?:,\d+)?p", sed_args[0]))


def is_forbidden_whole_file_cat(command: str) -> bool:
    """Return true for obvious unbounded whole-file `cat` reads."""
    try:
        tokens = _shell_tokens(command)
    except ValueError:
        return False

    command_separators = {";", "&&", "||"}
    i = 0
    while i < len(tokens):
        segment_end = i
        while segment_end < len(tokens) and tokens[segment_end] not in command_separators:
            segment_end += 1
        segment = tokens[i:segment_end]
        if segment and segment[0] == "cat":
            if any(token.startswith("<") or token.startswith(">") for token in segment):
                i = segment_end + 1
                continue
            if "|" in segment:
                pipe_index = segment.index("|")
                if len(segment[1:pipe_index]) > 0 and not _is_bounded_cat_pipe(segment, pipe_index):
                    return True
            elif len(segment) > 1:
                return True
        i = segment_end + 1
    return False


def forbidden_whole_file_cat_output() -> dict:
    return {"output": WHOLE_FILE_CAT_FORBIDDEN_OUTPUT, "returncode": 1, "exception_info": ""}


class AgentConfig(BaseModel):
    """Check the config files in minisweagent/config for example settings."""

    system_template: str
    """Template for the system message (the first message)."""
    instance_template: str
    """Template for the first user message specifying the task (the second message overall)."""
    step_limit: int = 0
    """Maximum number of steps the agent can take."""
    cost_limit: float = 3.0
    """Stop agent after exceeding (!) this cost."""
    wall_time_limit_seconds: int = 0
    """Stop agent after this many seconds of wall-clock time. 0 means no limit."""
    max_consecutive_format_errors: int = 3
    """Exit after this many format errors in a row (0 = no limit)."""
    output_path: Path | None = None
    """Save the trajectory to this path."""
    debug_exchange_path: Path | None = None
    """Append full model exchange debug events to this JSONL file."""
    debug_exchange_readable_path: Path | None = None
    """Append readable model exchange debug events to this Markdown file."""


class DefaultAgent:
    def __init__(self, model: Model, env: Environment, *, config_class: type = AgentConfig, **kwargs):
        """See the `AgentConfig` class for permitted keyword arguments."""
        self.config = config_class(**kwargs)
        self.messages: list[dict] = []
        self.model = model
        self.env = env
        self.extra_template_vars = {}
        self.logger = logging.getLogger("agent")
        self.cost = 0.0
        self.n_calls = 0
        self.n_consecutive_format_errors = 0
        self._debug_event_index = 0
        self._readable_exchange_index = 0
        self._start_time = time.time()

    def _write_debug_event(self, event: str, **data) -> None:
        if not self.config.debug_exchange_path:
            return
        self.config.debug_exchange_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "event": event,
            "event_index": self._debug_event_index,
            "timestamp": time.time(),
            "n_calls": self.n_calls,
            "active_message_count": len(self.messages),
            **data,
        }
        self._debug_event_index += 1
        mode = "w" if payload["event_index"] == 0 else "a"
        with self.config.debug_exchange_path.open(mode) as f:
            f.write(json.dumps(payload, default=str) + "\n")

    def _prepared_messages_for_debug(self, messages: list[dict]) -> list[dict] | None:
        prepare = getattr(self.model, "_prepare_messages_for_api", None)
        if not prepare:
            return None
        return prepare(messages)

    def _provider_exchange_for_debug(self) -> dict:
        return {
            "provider_request": getattr(self.model, "_last_provider_request", None),
            "provider_response": getattr(self.model, "_last_provider_response", None),
        }

    def _debug_markdown_text(self, value) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, list) and all(isinstance(item, dict) and "text" in item for item in value):
            return "\n\n".join(str(item["text"]) for item in value)
        return json.dumps(value, indent=2, default=str)

    def _is_tool_result_message(self, message: dict) -> bool:
        return (
            message.get("role") == "tool"
            or message.get("type") == "function_call_output"
            or (message.get("role") == "user" and "returncode" in self._debug_markdown_text(message.get("content")))
        )

    def _clip_tool_result_text(self, text: str, *, limit: int = 25) -> str:
        if "<output>" in text and "</output>" in text:
            before, rest = text.split("<output>", 1)
            body, after = rest.split("</output>", 1)
            prefix = before + "<output>"
            suffix = "</output>" + after
            body_prefix = "\n" if body.startswith("\n") else ""
            body_suffix = "\n" if body.endswith("\n") else ""
            clipped_body = self._clip_tool_result_text(body.strip("\n"), limit=limit)
            return prefix + body_prefix + clipped_body + body_suffix + suffix
        lines = text.splitlines()
        if len(lines) <= limit:
            return text
        clipped = "\n".join(lines[:limit])
        return f"{clipped}\n...[clipped after {limit} lines; {len(lines) - limit} more lines omitted]"

    def _render_debug_code_block(self, text: str, *, language: str = "text") -> str:
        fence = "```"
        while fence in text:
            fence += "`"
        return f"{fence}{language}\n{text}\n{fence}\n"

    def _render_debug_tool_calls(self, message: dict) -> str:
        rendered = []
        for i, tool_call in enumerate(message.get("tool_calls") or [], 1):
            rendered.append(self._render_debug_tool_call(tool_call, i))
        return "\n".join(rendered)

    def _render_debug_tool_call(self, tool_call: dict, index: int) -> str:
        function = tool_call.get("function", {})
        name = function.get("name") or tool_call.get("name") or "unknown"
        arguments = function.get("arguments", tool_call.get("arguments", ""))
        rendered = [f"Tool call {index}: `{name}`\n"]
        if isinstance(arguments, str):
            try:
                parsed_arguments = json.loads(arguments)
            except json.JSONDecodeError:
                parsed_arguments = {"arguments": arguments}
        else:
            parsed_arguments = arguments
        command = parsed_arguments.get("command") if isinstance(parsed_arguments, dict) else None
        if command is not None:
            rendered.append(self._render_debug_code_block(str(command), language="bash"))
        else:
            rendered.append(
                self._render_debug_code_block(json.dumps(parsed_arguments, indent=2, default=str), language="json")
            )
        return "\n".join(rendered)

    def _render_debug_response_output(self, message: dict) -> str:
        output_items = message.get("output")
        if not isinstance(output_items, list):
            return ""
        rendered = []
        for i, item in enumerate(output_items, 1):
            item_type = item.get("type", "output") if isinstance(item, dict) else "output"
            rendered.append(f"Output item {i}: `{item_type}`\n")
            if isinstance(item, dict) and item_type == "message":
                content = self._debug_markdown_text(item.get("content"))
                if content:
                    rendered.append(self._render_debug_code_block(content))
            elif isinstance(item, dict) and item_type == "function_call":
                rendered.append(self._render_debug_tool_call(item, i))
            else:
                rendered.append(self._render_debug_code_block(json.dumps(item, indent=2, default=str), language="json"))
        return "\n".join(rendered)

    def _render_debug_message(self, message: dict, index: int) -> str:
        role = message.get("role") or message.get("type") or "message"
        output = [f"#### Message {index}: `{role}`\n"]
        if "content" in message:
            content = self._debug_markdown_text(message.get("content"))
            if self._is_tool_result_message(message):
                content = self._clip_tool_result_text(content)
            if content:
                output.append(self._render_debug_code_block(content))
        if "output" in message and message.get("type") == "function_call_output":
            content = self._clip_tool_result_text(self._debug_markdown_text(message.get("output")))
            output.append(self._render_debug_code_block(content))
        elif "output" in message:
            rendered_output = self._render_debug_response_output(message)
            if rendered_output:
                output.append(rendered_output)
        tool_calls = self._render_debug_tool_calls(message)
        if tool_calls:
            output.append(tool_calls)
        extra_keys = [
            key for key in sorted(message) if key not in {"role", "type", "content", "output", "tool_calls", "extra"}
        ]
        if extra_keys:
            metadata = {key: message[key] for key in extra_keys}
            output.append(self._render_debug_code_block(json.dumps(metadata, indent=2, default=str), language="json"))
        return "\n".join(output).rstrip() + "\n"

    def _render_debug_messages(self, messages: list[dict]) -> str:
        return "\n".join(self._render_debug_message(message, i) for i, message in enumerate(messages, 1))

    def _render_debug_provider_payload(self, payload: dict) -> str:
        output = []
        for key, value in payload.items():
            output.append(f"#### `{key}`\n")
            if (
                key in {"messages", "input"}
                and isinstance(value, list)
                and all(isinstance(item, dict) for item in value)
            ):
                output.append(self._render_debug_messages(value))
            else:
                output.append(self._render_debug_code_block(json.dumps(value, indent=2, default=str), language="json"))
        return "\n".join(output).rstrip() + "\n"

    def _render_debug_provider_response(self, response) -> str:
        if not isinstance(response, dict):
            return self._render_debug_code_block(json.dumps(response, indent=2, default=str), language="json")
        output = []
        for key, value in response.items():
            output.append(f"#### `{key}`\n")
            if key == "message" and isinstance(value, dict):
                output.append(self._render_debug_message(value, 1))
            elif key == "choices" and isinstance(value, list):
                for i, choice in enumerate(value, 1):
                    output.append(f"Choice {i}\n")
                    message = choice.get("message") if isinstance(choice, dict) else None
                    if isinstance(message, dict):
                        output.append(self._render_debug_message(message, i))
                    else:
                        output.append(
                            self._render_debug_code_block(json.dumps(choice, indent=2, default=str), language="json")
                        )
            elif key == "output" and isinstance(value, list):
                output.append(self._render_debug_response_output({"output": value}))
            else:
                output.append(self._render_debug_code_block(json.dumps(value, indent=2, default=str), language="json"))
        return "\n".join(output).rstrip() + "\n"

    def _write_readable_debug(self, text: str, *, reset: bool = False) -> None:
        if not self.config.debug_exchange_readable_path:
            return
        self.config.debug_exchange_readable_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "w" if reset else "a"
        with self.config.debug_exchange_readable_path.open(mode) as f:
            f.write(text)

    def _write_readable_provider_request(
        self, event: str, provider: str, payload: dict, *, input_tokens: int
    ) -> int | None:
        if not self.config.debug_exchange_readable_path:
            return None
        exchange_index = self._readable_exchange_index
        self._readable_exchange_index += 1
        reset = exchange_index == 0
        header = "# mini-swe-agent readable debug exchanges\n\n" if reset else ""
        exchange_separator = "" if reset else "------------------------------\n\n"
        self._write_readable_debug(
            header
            + exchange_separator
            + f"## Exchange {exchange_index}: `{event}`\n\n"
            + f"- model_call: {self.n_calls}\n"
            + f"- provider: {provider}\n"
            + f"- input_tokens_estimate: {input_tokens}\n\n"
            + "### Sent to provider\n\n"
            + self._render_debug_provider_payload(payload)
            + "\n",
            reset=reset,
        )
        return exchange_index

    def _write_readable_provider_response(self, exchange_index: int | None, response=None, *, error: str = "") -> None:
        if not self.config.debug_exchange_readable_path or exchange_index is None:
            return
        if error:
            body = "### Provider error\n\n" + self._render_debug_code_block(error)
        else:
            body = "### Provider response\n\n" + self._render_debug_provider_response(response)
        self._write_readable_debug(body + "\n")

    def _readable_debug_callback(self, event: str, *, input_tokens: int):
        exchange_index = None

        def callback(phase: str, provider: str, payload=None, response=None, error: str = "") -> None:
            nonlocal exchange_index
            if phase == "request":
                callback.request_logged = True
                exchange_index = self._write_readable_provider_request(
                    event, provider, payload or {}, input_tokens=input_tokens
                )
            elif phase == "response":
                self._write_readable_provider_response(exchange_index, response=response)
            elif phase == "error":
                self._write_readable_provider_response(exchange_index, error=error)

        callback.request_logged = False
        return callback

    def _assert_readable_debug_callback_used(self, callback) -> None:
        if self.config.debug_exchange_readable_path and not getattr(callback, "request_logged", False):
            raise RuntimeError(
                "Readable debug exchanges require the model adapter to report the exact provider request payload."
            )

    def get_template_vars(self, **kwargs) -> dict:
        return recursive_merge(
            self.config.model_dump(),
            self.env.get_template_vars(),
            self.model.get_template_vars(),
            {
                "n_model_calls": self.n_calls,
                "model_cost": self.cost,
                "elapsed_seconds": int(time.time() - self._start_time),
            },
            self.extra_template_vars,
            kwargs,
        )

    def _render_template(self, template: str) -> str:
        return Template(template, undefined=StrictUndefined).render(**self.get_template_vars())

    def add_messages(self, *messages: dict) -> list[dict]:
        self.logger.debug(messages)  # set log level to debug to see
        self.messages.extend(messages)
        return list(messages)

    def _estimate_tokens(self, messages: list[dict]) -> int:
        """Cheap token estimate for context management."""
        return max(1, len(json.dumps(messages, default=str)) // 4)

    def _estimate_api_tokens(self, messages: list[dict]) -> int:
        """Estimate tokens for the payload shape that will actually be sent to the model."""
        prepared = self._prepared_messages_for_debug(messages)
        if prepared is None:
            prepared = [{k: v for k, v in msg.items() if k != "extra"} for msg in messages]
        return self._estimate_tokens(prepared)

    def _context_limit(self) -> int:
        config = getattr(self.model, "config", None)
        model_kwargs = getattr(config, "model_kwargs", {}) or {}
        for key in ("max_input_tokens", "context_window", "context_limit", "max_context_tokens"):
            value = model_kwargs.get(key) or getattr(config, key, None)
            if value:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    pass
        if value := os.getenv("MAX_INPUT_TOKENS"):
            return int(value)
        return 0

    def _context_compaction_fraction(self, env_key: str, default: float) -> float:
        value = os.getenv(env_key)
        if not value:
            return default
        fraction = float(value)
        if fraction > 1:
            fraction /= 100
        if not 0 < fraction < 1:
            raise ValueError(f"{env_key} must be between 0 and 1, or between 0 and 100 as a percentage")
        return fraction

    def _is_observation_for_previous_action(self, message: dict) -> bool:
        return (
            message.get("role") == "tool"
            or message.get("type") == "function_call_output"
            or (message.get("role") == "user" and "returncode" in get_content_string(message))
        )

    def _recent_message_groups(self, messages: list[dict]) -> list[list[dict]]:
        """Group messages so an assistant action and its observation stay together."""
        groups = []
        i = len(messages) - 1
        while i >= 2:
            message = messages[i]
            if (
                self._is_observation_for_previous_action(message)
                and i > 2
                and messages[i - 1].get("extra", {}).get("actions")
            ):
                groups.append(messages[i - 1 : i + 1])
                i -= 2
            else:
                groups.append([message])
                i -= 1
        return groups

    def _choose_tail_start(self, messages: list[dict], token_budget: int) -> int:
        groups = self._recent_message_groups(messages)
        if not groups:
            return len(messages)
        tail = []
        for group in groups:
            candidate = group + tail
            if self._estimate_api_tokens(candidate) > token_budget:
                if not tail:
                    raise RuntimeError(
                        "Newest tail message group does not fit in the configured compaction tail budget."
                    )
                break
            tail = candidate
        return len(messages) - len(tail)

    def _api_messages(self, messages: list[dict]) -> list[dict]:
        prepared = self._prepared_messages_for_debug(messages)
        if prepared is None:
            return [{k: v for k, v in msg.items() if k != "extra"} for msg in messages]
        return prepared

    def _message_groups(self, messages: list[dict]) -> list[list[dict]]:
        """Group messages so an assistant action and its observation stay together."""
        groups = []
        i = 0
        while i < len(messages):
            if (
                i + 1 < len(messages)
                and messages[i].get("extra", {}).get("actions")
                and self._is_observation_for_previous_action(messages[i + 1])
            ):
                groups.append(messages[i : i + 2])
                i += 2
            else:
                groups.append([messages[i]])
                i += 1
        return groups

    def _compaction_summary_messages(self, messages: list[dict], token_budget: int) -> list[dict]:
        messages_json = json.dumps(self._api_messages(messages), separators=(",", ":"), default=str)
        prompt = (
            "Summarize a mini-swe-agent run so the same run can continue. "
            "Do not call tools. Preserve enough concrete detail for the next model call to continue safely. "
            "Include these headings: "
            "current objective, files inspected, files modified, commands run, important test results, "
            "failed approaches, current plan, remaining TODOs, important facts that must not be forgotten.\n\n"
            f"Target length: {token_budget} tokens. This is a target, not a hard maximum; "
            "do not omit important facts just to make the summary shorter.\n\n"
            f"Messages to summarize:\n{messages_json}"
        )
        return [
            {"role": "system", "content": "You write concise state summaries for continuing software-agent runs."},
            {"role": "user", "content": prompt},
        ]

    def _query_compaction_summary(
        self, messages: list[dict], token_budget: int, context_limit: int, *, event: str
    ) -> str:
        summary_messages = self._compaction_summary_messages(messages, token_budget)
        request_tokens = self._estimate_api_tokens(summary_messages)
        if request_tokens + token_budget > context_limit:
            raise RuntimeError(
                "Compaction summary request exceeds context limit: "
                f"input={request_tokens}, output={token_budget}, limit={context_limit}"
            )
        raw_query = getattr(self.model, "_query", None)
        request_kwargs = {"max_tokens": token_budget, "tool_choice": "none"}
        text_query = getattr(self.model, "query_text", None)
        if text_query:
            prepared_messages = self._prepared_messages_for_debug(summary_messages)
            readable_debug_callback = self._readable_debug_callback(event, input_tokens=request_tokens)
            response = text_query(
                summary_messages, max_tokens=token_budget, readable_debug_callback=readable_debug_callback
            )
            self._assert_readable_debug_callback_used(readable_debug_callback)
            if hasattr(response, "model_dump"):
                response = response.model_dump()
            if isinstance(response, dict) and response.get("choices"):
                summary = get_content_string(response["choices"][0].get("message", {}))
            else:
                summary = get_content_string(response if isinstance(response, dict) else {"content": str(response)})
            self._write_debug_event(
                event,
                request_messages=summary_messages,
                prepared_messages=prepared_messages,
                input_tokens=request_tokens,
                request_kwargs={"max_tokens": token_budget},
                raw_response=response,
                **self._provider_exchange_for_debug(),
                summary=summary,
            )
            return summary
        if raw_query:
            prepare = getattr(self.model, "_prepare_messages_for_api", lambda messages: messages)
            prepared_messages = prepare(summary_messages)
            readable_debug_callback = self._readable_debug_callback(event, input_tokens=request_tokens)
            try:
                response = raw_query(
                    prepared_messages, **request_kwargs, readable_debug_callback=readable_debug_callback
                )
            except TypeError:
                request_kwargs = {"max_tokens": token_budget}
                response = raw_query(
                    prepared_messages, max_tokens=token_budget, readable_debug_callback=readable_debug_callback
                )
            self._assert_readable_debug_callback_used(readable_debug_callback)
            if hasattr(response, "model_dump"):
                response = response.model_dump()
            if isinstance(response, dict) and response.get("choices"):
                summary = get_content_string(response["choices"][0].get("message", {}))
            else:
                summary = get_content_string(response if isinstance(response, dict) else {"content": str(response)})
            self._write_debug_event(
                event,
                request_messages=summary_messages,
                prepared_messages=prepared_messages,
                input_tokens=request_tokens,
                request_kwargs=request_kwargs,
                raw_response=response,
                **self._provider_exchange_for_debug(),
                summary=summary,
            )
            return summary
        readable_debug_callback = self._readable_debug_callback(event, input_tokens=request_tokens)
        message = self.model.query(summary_messages, readable_debug_callback=readable_debug_callback)
        self._assert_readable_debug_callback_used(readable_debug_callback)
        summary = get_content_string(message)
        self._write_debug_event(
            event,
            request_messages=summary_messages,
            prepared_messages=None,
            input_tokens=request_tokens,
            request_kwargs={},
            response_message=message,
            **self._provider_exchange_for_debug(),
            summary=summary,
        )
        return summary

    def _summarize_bounded(self, messages: list[dict], token_budget: int, context_limit: int, *, depth: int = 0) -> str:
        if (
            self._estimate_api_tokens(self._compaction_summary_messages(messages, token_budget)) + token_budget
            <= context_limit
        ):
            event = "compaction_final_summary_call" if depth else "compaction_summary_call"
            return self._query_compaction_summary(messages, token_budget, context_limit, event=event)

        input_budget = context_limit - token_budget
        if input_budget <= 0:
            raise RuntimeError("Configured summary target leaves no room for compaction input.")

        chunks: list[list[dict]] = []
        current: list[dict] = []
        for group in self._message_groups(messages):
            candidate = current + group
            if self._estimate_api_tokens(self._compaction_summary_messages(candidate, token_budget)) <= input_budget:
                current = candidate
                continue
            if not current:
                raise RuntimeError(
                    "Single middle message group does not fit in the configured compaction input budget."
                )
            chunks.append(current)
            current = group
            if self._estimate_api_tokens(self._compaction_summary_messages(current, token_budget)) > input_budget:
                raise RuntimeError(
                    "Single middle message group does not fit in the configured compaction input budget."
                )
        if current:
            chunks.append(current)

        chunk_summary_messages = []
        for i, chunk in enumerate(chunks):
            input_tokens = self._estimate_api_tokens(self._compaction_summary_messages(chunk, token_budget))
            self._write_debug_event(
                "compaction_chunk_planned",
                chunk_index=i,
                chunk_message_count=len(chunk),
                input_tokens=input_tokens,
                output_target=token_budget,
            )
            summary = self._query_compaction_summary(
                chunk, token_budget, context_limit, event="compaction_chunk_summary_call"
            )
            chunk_summary_messages.append(
                self.model.format_message(
                    role="user",
                    content=f'<compact_chunk_summary index="{i}">\n{summary.strip()}\n</compact_chunk_summary>',
                )
            )
        return self._summarize_bounded(chunk_summary_messages, token_budget, context_limit, depth=depth + 1)

    def _maybe_compact_messages(self) -> None:
        limit = self._context_limit()
        if not limit:
            return
        before = self._estimate_api_tokens(self.messages)
        trigger_fraction = self._context_compaction_fraction("MSWEA_CONTEXT_COMPACT_AT", 2 / 3)
        target_fraction = self._context_compaction_fraction("MSWEA_CONTEXT_COMPACT_TO", 1 / 3)
        tail_fraction = self._context_compaction_fraction("MSWEA_CONTEXT_TAIL_TARGET_PERCENT", 0.5)
        threshold = int(limit * trigger_fraction)
        target = int(limit * target_fraction)
        tail_target = int(target * tail_fraction)
        compaction_logger.info(
            "Considering context compaction: estimated_tokens=%s limit=%s threshold=%s",
            before,
            limit,
            threshold,
        )
        self._write_debug_event(
            "compaction_considered",
            estimated_tokens=before,
            context_limit=limit,
            threshold=threshold,
            target=target,
            tail_target=tail_target,
            compact_at=trigger_fraction,
            compact_to=target_fraction,
            tail_target_percent=tail_fraction,
        )
        if before < threshold:
            return
        if len(self.messages) <= 6:
            compaction_logger.info("Skipping context compaction: message history is too small to benefit")
            return
        head = self.messages[:2]
        head_tokens = self._estimate_api_tokens(head)
        if head_tokens >= target:
            raise RuntimeError("Compaction head does not fit in the configured compact-to target.")
        tail_start = self._choose_tail_start(self.messages, token_budget=tail_target)
        tail = self.messages[tail_start:]
        tail_tokens = self._estimate_api_tokens(tail)
        summary_target = target - head_tokens - tail_tokens
        if summary_target <= 0:
            raise RuntimeError(
                "Configured compaction target leaves no room for a summary after preserving head and tail."
            )
        middle = self.messages[2:tail_start]
        if not middle:
            compaction_logger.info("Skipping context compaction: no older middle history to summarize")
            return
        compaction_logger.info("Triggering context compaction: estimated_tokens=%s target_tokens=%s", before, target)
        self._write_debug_event(
            "compaction_triggered",
            estimated_tokens=before,
            context_limit=limit,
            threshold=threshold,
            target=target,
            tail_target=tail_target,
            head_tokens=head_tokens,
            tail_tokens=tail_tokens,
            summary_budget=summary_target,
            head_message_count=len(head),
            middle_message_count=len(middle),
            tail_message_count=len(tail),
            tail_start=tail_start,
            middle_messages=middle,
        )
        summary = self._summarize_bounded(middle, summary_target, limit)
        summary_message = self.model.format_message(
            role="user",
            content="<compact_summary>\n" + summary.strip() + "\n</compact_summary>",
            extra={"compact_summary": True},
        )
        self.messages = head + [summary_message] + tail
        after = self._estimate_api_tokens(self.messages)
        compaction_logger.info(
            "Finished context compaction: estimated_tokens_before=%s estimated_tokens_after=%s",
            before,
            after,
        )
        self._write_debug_event(
            "compaction_finished",
            estimated_tokens_before=before,
            estimated_tokens_after=after,
            active_messages=self.messages,
        )

    def handle_uncaught_exception(self, e: Exception) -> list[dict]:
        return self.add_messages(
            self.model.format_message(
                role="exit",
                content=str(e),
                extra={
                    "exit_status": type(e).__name__,
                    "submission": "",
                    "exception_str": str(e),
                    "traceback": traceback.format_exc(),
                },
            )
        )

    def run(self, task: str = "", **kwargs) -> dict:
        """Run step() until agent is finished. Returns dictionary with exit_status, submission keys."""
        self.extra_template_vars |= {"task": task, **kwargs}
        self.messages = []
        self.add_messages(
            self.model.format_message(role="system", content=self._render_template(self.config.system_template)),
            self.model.format_message(role="user", content=self._render_template(self.config.instance_template)),
        )
        self._write_debug_event(
            "run_start",
            task=task,
            kwargs=kwargs,
            agent_config=self.config.model_dump(mode="json"),
            model_config=self.model.serialize(),
            environment_config=self.env.serialize(),
            active_messages=self.messages,
        )
        while True:
            try:
                self.step()
                self.n_consecutive_format_errors = 0  # reset on any clean step
            except FormatError as e:
                # The call was billed before parsing failed, so query() never got to charge it.
                self.cost += e.messages[0].get("extra", {}).get("cost", 0.0)
                self.n_consecutive_format_errors += 1
                if 0 < self.config.max_consecutive_format_errors <= self.n_consecutive_format_errors:
                    self.add_messages(
                        *e.messages,
                        {
                            "role": "exit",
                            "content": "RepeatedFormatError",
                            "extra": {"exit_status": "RepeatedFormatError", "submission": ""},
                        },
                    )
                else:
                    self.add_messages(*e.messages)
            except InterruptAgentFlow as e:
                self.add_messages(*e.messages)
            except Exception as e:
                self.handle_uncaught_exception(e)
                raise
            finally:
                self.save(self.config.output_path)
            if self.messages[-1].get("role") == "exit":
                break
        self._write_debug_event("run_end", result=self.messages[-1].get("extra", {}), active_messages=self.messages)
        return self.messages[-1].get("extra", {})

    def step(self) -> list[dict]:
        """Query the LM, execute actions."""
        return self.execute_actions(self.query())

    def query(self) -> dict:
        """Query the model and return model messages. Override to add hooks."""
        if 0 < self.config.step_limit <= self.n_calls or 0 < self.config.cost_limit <= self.cost:
            raise LimitsExceeded(
                {
                    "role": "exit",
                    "content": "LimitsExceeded",
                    "extra": {"exit_status": "LimitsExceeded", "submission": ""},
                }
            )
        if 0 < self.config.wall_time_limit_seconds <= int(time.time() - self._start_time):
            raise TimeExceeded(
                {
                    "role": "exit",
                    "content": "TimeExceeded",
                    "extra": {"exit_status": "TimeExceeded", "submission": ""},
                }
            )
        self._maybe_compact_messages()
        self.n_calls += 1
        request_messages = list(self.messages)
        prepared_messages = self._prepared_messages_for_debug(request_messages)
        limit = self._context_limit()
        request_tokens = self._estimate_api_tokens(request_messages)
        if limit and request_tokens > limit:
            raise RuntimeError(f"Model request exceeds context limit: input={request_tokens}, limit={limit}")
        readable_debug_callback = self._readable_debug_callback("model_call", input_tokens=request_tokens)
        try:
            message = self.model.query(self.messages, readable_debug_callback=readable_debug_callback)
        except Exception as e:
            self._write_debug_event(
                "model_call",
                request_messages=request_messages,
                prepared_messages=prepared_messages,
                input_tokens=request_tokens,
                **self._provider_exchange_for_debug(),
                error=repr(e),
            )
            raise
        self._assert_readable_debug_callback_used(readable_debug_callback)
        self.cost += message.get("extra", {}).get("cost", 0.0)
        self._write_debug_event(
            "model_call",
            request_messages=request_messages,
            prepared_messages=prepared_messages,
            input_tokens=request_tokens,
            response_message=message,
            raw_response=message.get("extra", {}).get("response"),
            **self._provider_exchange_for_debug(),
            usage=(message.get("extra", {}).get("response") or {}).get("usage")
            if isinstance(message.get("extra", {}).get("response"), dict)
            else None,
            cost=message.get("extra", {}).get("cost", 0.0),
        )
        self.add_messages(message)
        return message

    def execute_actions(self, message: dict) -> list[dict]:
        """Execute actions in message, add observation messages, return them."""
        actions = message.get("extra", {}).get("actions", [])
        outputs = []
        try:
            for action in actions:
                outputs.append(self._execute_action(action))
        finally:
            observation_messages = self.model.format_observation_messages(message, outputs, self.get_template_vars())
            self._write_debug_event(
                "action_execution",
                actions=actions,
                outputs=outputs,
                observation_messages=observation_messages,
            )
        return self.add_messages(*observation_messages)

    def _execute_action(self, action: dict) -> dict:
        if is_forbidden_whole_file_cat(action.get("command", "")):
            return forbidden_whole_file_cat_output()
        return self.env.execute(action)

    def serialize(self, *extra_dicts) -> dict:
        """Serialize agent state to a json-compatible nested dictionary for saving."""
        last_message = self.messages[-1] if self.messages else {}
        last_extra = last_message.get("extra", {})
        agent_data = {
            "info": {
                "model_stats": {
                    "instance_cost": self.cost,
                    "api_calls": self.n_calls,
                },
                "config": {
                    "agent": self.config.model_dump(mode="json"),
                    "agent_type": f"{self.__class__.__module__}.{self.__class__.__name__}",
                },
                "mini_version": __version__,
                "exit_status": last_extra.get("exit_status", ""),
                "submission": last_extra.get("submission", ""),
            },
            "messages": self.messages,
            "trajectory_format": "mini-swe-agent-1.1",
        }
        return recursive_merge(agent_data, self.model.serialize(), self.env.serialize(), *extra_dicts)

    def save(self, path: Path | None, *extra_dicts) -> dict:
        """Save the trajectory of the agent to a file if path is given. Returns full serialized data.
        You can pass additional dictionaries with extra data to be (recursively) merged into the output data.
        """
        data = self.serialize(*extra_dicts)
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2))
        return data
