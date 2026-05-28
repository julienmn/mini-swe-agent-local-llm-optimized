"""Basic agent class. See https://mini-swe-agent.com/latest/advanced/control_flow/ for visual explanation
or https://minimal-agent.com for a tutorial on the basic building principles.
"""

import json
import logging
import os
import time
import traceback
from pathlib import Path

from jinja2 import StrictUndefined, Template
from pydantic import BaseModel

from minisweagent import Environment, Model, __version__
from minisweagent.exceptions import InterruptAgentFlow, LimitsExceeded, TimeExceeded
from minisweagent.models.utils.content_string import get_content_string
from minisweagent.utils.serialize import recursive_merge

compaction_logger = logging.getLogger("minisweagent.agent.compaction")


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
    output_path: Path | None = None
    """Save the trajectory to this path."""
    debug_exchange_path: Path | None = None
    """Append full model exchange debug events to this JSONL file."""


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
        self._debug_event_index = 0
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
                    raise RuntimeError("Newest tail message group does not fit in the configured compaction tail budget.")
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
            "Summarize the older middle of a mini-swe-agent run so the same run can continue. "
            "Do not call tools. Preserve enough concrete detail for the next model call to continue safely. "
            "Include these headings: "
            "current objective, user constraints, files inspected, files modified, commands run, test results, "
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
            response = text_query(summary_messages, max_tokens=token_budget)
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
                summary=summary,
            )
            return summary
        if raw_query:
            prepare = getattr(self.model, "_prepare_messages_for_api", lambda messages: messages)
            prepared_messages = prepare(summary_messages)
            try:
                response = raw_query(prepared_messages, **request_kwargs)
            except TypeError:
                request_kwargs = {"max_tokens": token_budget}
                response = raw_query(prepared_messages, max_tokens=token_budget)
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
                summary=summary,
            )
            return summary
        message = self.model.query(summary_messages)
        summary = get_content_string(message)
        self._write_debug_event(
            event,
            request_messages=summary_messages,
            prepared_messages=None,
            input_tokens=request_tokens,
            request_kwargs={},
            response_message=message,
            summary=summary,
        )
        return summary

    def _summarize_bounded(
        self, messages: list[dict], token_budget: int, context_limit: int, *, depth: int = 0
    ) -> str:
        if self._estimate_api_tokens(self._compaction_summary_messages(messages, token_budget)) + token_budget <= context_limit:
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
                raise RuntimeError("Single middle message group does not fit in the configured compaction input budget.")
            chunks.append(current)
            current = group
            if self._estimate_api_tokens(self._compaction_summary_messages(current, token_budget)) > input_budget:
                raise RuntimeError("Single middle message group does not fit in the configured compaction input budget.")
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
                    content=f"<compact_chunk_summary index=\"{i}\">\n{summary.strip()}\n</compact_chunk_summary>",
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
            raise RuntimeError("Configured compaction target leaves no room for a summary after preserving head and tail.")
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
        try:
            message = self.model.query(self.messages)
        except Exception as e:
            self._write_debug_event(
                "model_call",
                request_messages=request_messages,
                prepared_messages=prepared_messages,
                input_tokens=request_tokens,
                error=repr(e),
            )
            raise
        self.cost += message.get("extra", {}).get("cost", 0.0)
        self._write_debug_event(
            "model_call",
            request_messages=request_messages,
            prepared_messages=prepared_messages,
            input_tokens=request_tokens,
            response_message=message,
            raw_response=message.get("extra", {}).get("response"),
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
                outputs.append(self.env.execute(action))
        finally:
            observation_messages = self.model.format_observation_messages(message, outputs, self.get_template_vars())
            self._write_debug_event(
                "action_execution",
                actions=actions,
                outputs=outputs,
                observation_messages=observation_messages,
            )
        return self.add_messages(*observation_messages)

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
