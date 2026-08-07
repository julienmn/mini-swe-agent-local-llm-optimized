import json
from pathlib import Path

import pytest
import yaml

from minisweagent.agents.default import (
    WHOLE_FILE_CAT_FORBIDDEN_OUTPUT,
    DefaultAgent,
    is_forbidden_whole_file_cat,
)
from minisweagent.environments.local import LocalEnvironment
from minisweagent.exceptions import FormatError
from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.test_models import (
    DeterministicModel,
    DeterministicResponseAPIToolcallModel,
    DeterministicToolcallModel,
    make_output,
    make_response_api_output,
    make_toolcall_output,
)
from minisweagent.models.ollama_model import OllamaModel

# --- Helper functions to abstract message format differences ---


def get_text(msg: dict) -> str:
    """Extract text content from a message regardless of format."""
    content = msg.get("content")
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list) and content:
        return content[0].get("text", "")
    return ""


def get_observation_text(msg: dict) -> str:
    """Extract observation text from a message (handles all formats)."""
    if msg.get("type") == "function_call_output":
        return msg.get("output", "")
    return get_text(msg)


def is_assistant_message(msg: dict) -> bool:
    """Check if message is an assistant/response message."""
    return msg.get("role") == "assistant" or msg.get("object") == "response"


def is_observation_message(msg: dict) -> bool:
    """Check if message is an observation message."""
    if msg.get("type") == "function_call_output":
        return True
    if msg.get("role") == "tool":
        return True
    if msg.get("role") == "user" and "returncode" in get_text(msg):
        return True
    return False


# --- Fixtures ---


@pytest.fixture
def default_config():
    """Load default agent config from config/default.yaml"""
    config_path = Path("src/minisweagent/config/default.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config["agent"]


@pytest.fixture
def toolcall_config():
    """Load toolcall agent config from config/mini.yaml"""
    config_path = Path("src/minisweagent/config/mini.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config["agent"]


def minimal_agent_config(**kwargs):
    return {
        "system_template": "sys",
        "instance_template": "{{task}}",
        "cost_limit": 10.0,
        **kwargs,
    }


def make_text_model(outputs_spec: list[tuple[str, list[dict]]], **kwargs) -> DeterministicModel:
    """Create a DeterministicModel from a list of (content, actions) tuples."""
    return DeterministicModel(outputs=[make_output(content, actions) for content, actions in outputs_spec], **kwargs)


class DebugProviderDeterministicModel(DeterministicModel):
    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        self._last_provider_request = {"provider": "debug-test", "messages": messages, "kwargs": kwargs}
        output = super().query(messages, **kwargs)
        self._last_provider_response = {"message": output}
        return output


class PreparedDeterministicModel(DeterministicModel):
    def _prepare_messages_for_api(self, messages: list[dict]) -> list[dict]:
        return [{key: value for key, value in message.items() if key != "extra"} for message in messages]

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        readable_debug_callback = kwargs.pop("readable_debug_callback", None)
        payload = {"model": self.config.model_name, "messages": self._prepare_messages_for_api(messages)}
        if readable_debug_callback:
            readable_debug_callback("request", provider="deterministic", payload=payload)
        try:
            response = super().query(messages, **kwargs)
        except Exception as e:
            if readable_debug_callback:
                readable_debug_callback("error", provider="deterministic", error=repr(e))
            raise
        if readable_debug_callback:
            readable_debug_callback("response", provider="deterministic", response=response)
        return response


class AssertingReadableRequestWrittenModel(PreparedDeterministicModel):
    def __init__(self, *, readable_path: Path, **kwargs):
        super().__init__(**kwargs)
        self.readable_path = readable_path

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        readable_debug_callback = kwargs.pop("readable_debug_callback", None)
        payload = {"model": self.config.model_name, "messages": self._prepare_messages_for_api(messages)}
        if readable_debug_callback:
            readable_debug_callback("request", provider="deterministic", payload=payload)
        assert self.readable_path.exists()
        text = self.readable_path.read_text()
        assert "### Sent to provider" in text
        assert "### Provider response" not in text
        response = super(PreparedDeterministicModel, self).query(messages, **kwargs)
        if readable_debug_callback:
            readable_debug_callback("response", provider="deterministic", response=response)
        return response


class RecordingEnvironment:
    def __init__(self):
        self.actions = []

    def execute(self, action: dict) -> dict:
        self.actions.append(action)
        return {"output": "executed", "returncode": 0, "exception_info": ""}

    def get_template_vars(self, **kwargs) -> dict:
        return kwargs


def make_tc_model(outputs_spec: list[tuple[str, list[dict]]], **kwargs) -> DeterministicToolcallModel:
    """Create a DeterministicToolcallModel from a list of (content, actions) tuples."""
    outputs = []
    for i, (content, actions) in enumerate(outputs_spec):
        tc_actions = []
        tool_calls = []
        for j, action in enumerate(actions):
            tool_call_id = f"call_{i}_{j}"
            tc_actions.append({"command": action["command"], "tool_call_id": tool_call_id})
            tool_calls.append(
                {
                    "id": tool_call_id,
                    "type": "function",
                    "function": {"name": "bash", "arguments": f'{{"command": "{action["command"]}"}}'},
                }
            )
        outputs.append(make_toolcall_output(content, tool_calls, tc_actions))
    return DeterministicToolcallModel(outputs=outputs, **kwargs)


def make_response_api_model(
    outputs_spec: list[tuple[str, list[dict]]], **kwargs
) -> DeterministicResponseAPIToolcallModel:
    """Create a DeterministicResponseAPIToolcallModel from a list of (content, actions) tuples."""
    outputs = []
    for i, (content, actions) in enumerate(outputs_spec):
        api_actions = []
        for j, action in enumerate(actions):
            tool_call_id = f"call_resp_{i}_{j}"
            api_actions.append({"command": action["command"], "tool_call_id": tool_call_id})
        outputs.append(make_response_api_output(content, api_actions))
    return DeterministicResponseAPIToolcallModel(outputs=outputs, **kwargs)


@pytest.fixture(params=["text", "toolcall", "response_api"])
def model_factory(request, default_config, toolcall_config):
    """Parametrized fixture that returns (factory_fn, config) for all three model types."""
    if request.param == "text":
        return make_text_model, default_config
    elif request.param == "toolcall":
        return make_tc_model, toolcall_config
    else:  # response_api
        return make_response_api_model, toolcall_config


# --- Tests ---


@pytest.mark.parametrize(
    "command",
    [
        "cat src/foo.py",
        "cat ./src/foo.py",
        "cat a.py b.py",
    ],
)
def test_whole_file_cat_guard_rejects_obvious_reads(command):
    assert is_forbidden_whole_file_cat(command)


@pytest.mark.parametrize(
    "command",
    [
        "cat <<'EOF' > newfile.py",
        "cat src/foo.py | head -50",
        "cat src/foo.py|head -50",
        "cat ./src/listening_room/user_text_overlay.py | sed -n '532,570p'",
    ],
)
def test_whole_file_cat_guard_allows_bounded_or_write_uses(command):
    assert not is_forbidden_whole_file_cat(command)


def test_default_agent_rejects_whole_file_cat_without_executing():
    env = RecordingEnvironment()
    agent = DefaultAgent(
        model=make_text_model([]),
        env=env,
        **minimal_agent_config(),
    )
    message = make_output("Read file", [{"command": "cat src/foo.py"}])

    observations = agent.execute_actions(message)

    assert env.actions == []
    assert len(observations) == 1
    assert WHOLE_FILE_CAT_FORBIDDEN_OUTPUT in get_observation_text(observations[0])


@pytest.mark.parametrize(
    ("line_count", "expected_returncode", "expected_output"),
    [
        (50, 0, "line 50"),
        (51, 1, "Output of files larger than 50 lines is forbidden."),
    ],
)
def test_default_agent_limits_whole_file_reads_by_line_count(
    tmp_path, line_count, expected_returncode, expected_output
):
    path = tmp_path / "file.txt"
    path.write_text("".join(f"line {i}\n" for i in range(1, line_count + 1)))
    agent = DefaultAgent(
        model=make_text_model([]),
        env=LocalEnvironment(),
        **minimal_agent_config(whole_file_read_max_lines=50),
    )

    observations = agent.execute_actions(make_output("Read file", [{"command": f"cat {path}"}]))

    assert f"<returncode>{expected_returncode}</returncode>" in get_observation_text(observations[0])
    assert expected_output in get_observation_text(observations[0])


def test_successful_completion(model_factory):
    """Test agent completes successfully when COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT is encountered."""
    factory, config = model_factory
    agent = DefaultAgent(
        model=factory(
            [
                ("I'll echo a message", [{"command": "echo 'hello world'"}]),
                (
                    "Now finishing",
                    [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'Task completed successfully'"}],
                ),
            ]
        ),
        env=LocalEnvironment(),
        **config,
    )

    info = agent.run("Echo hello world then finish")
    assert info["exit_status"] == "Submitted"
    assert info["submission"] == "Task completed successfully\n"
    assert agent.n_calls == 2


def test_step_limit_enforcement(model_factory):
    """Test agent stops when step limit is reached."""
    factory, config = model_factory
    agent = DefaultAgent(
        model=factory(
            [
                ("First command", [{"command": "echo 'step1'"}]),
                ("Second command", [{"command": "echo 'step2'"}]),
            ]
        ),
        env=LocalEnvironment(),
        **{**config, "step_limit": 1},
    )

    info = agent.run("Run multiple commands")
    assert info["exit_status"] == "LimitsExceeded"
    assert agent.n_calls == 1


def test_cost_limit_enforcement(model_factory):
    """Test agent stops when cost limit is reached."""
    factory, config = model_factory
    agent = DefaultAgent(
        model=factory([("Test", [{"command": "echo 'test'"}])]),
        env=LocalEnvironment(),
        **{**config, "cost_limit": 0.5},
    )

    info = agent.run("Test cost limit")
    assert info["exit_status"] == "LimitsExceeded"


def test_timeout_handling(model_factory):
    """Test agent handles command timeouts properly."""
    factory, config = model_factory
    agent = DefaultAgent(
        model=factory(
            [
                ("Long sleep", [{"command": "sleep 5"}]),  # This will timeout
                ("Quick finish", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'recovered'"}]),
            ]
        ),
        env=LocalEnvironment(timeout=1),  # Very short timeout
        **config,
    )

    info = agent.run("Test timeout handling")
    assert info["exit_status"] == "Submitted"
    assert info["submission"] == "recovered\n"
    # Should have timeout error message in observation
    timed_out = [msg for msg in agent.messages if "timed out" in get_observation_text(msg)]
    assert len(timed_out) == 1


def test_timeout_captures_partial_output(model_factory):
    """Test that timeout error captures partial output from commands that produce output before timing out."""
    factory, config = model_factory
    num1, num2 = 111, 9
    calculation_command = f"echo $(({num1}*{num2})); sleep 10"
    expected_output = str(num1 * num2)
    agent = DefaultAgent(
        model=factory(
            [
                ("Output then sleep", [{"command": calculation_command}]),
                ("Quick finish", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'recovered'"}]),
            ]
        ),
        env=LocalEnvironment(timeout=1),
        **config,
    )
    info = agent.run("Test timeout with partial output")
    assert info["exit_status"] == "Submitted"
    assert info["submission"] == "recovered\n"
    timed_out = [msg for msg in agent.messages if "timed out" in get_observation_text(msg)]
    assert len(timed_out) == 1
    assert expected_output in get_observation_text(timed_out[0])


def test_multiple_steps_before_completion(model_factory):
    """Test agent can handle multiple steps before finding completion signal."""
    factory, config = model_factory
    agent = DefaultAgent(
        model=factory(
            [
                ("Step 1", [{"command": "echo 'first'"}]),
                ("Step 2", [{"command": "echo 'second'"}]),
                ("Step 3", [{"command": "echo 'third'"}]),
                (
                    "Final step",
                    [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'completed all steps'"}],
                ),
            ]
        ),
        env=LocalEnvironment(),
        **{**config, "cost_limit": 5.0},  # Increase cost limit to allow all 4 calls
    )

    info = agent.run("Multi-step task")
    assert info["exit_status"] == "Submitted"
    assert info["submission"] == "completed all steps\n"
    assert agent.n_calls == 4


def test_custom_config(model_factory):
    """Test agent works with custom configuration."""
    factory, config = model_factory
    agent = DefaultAgent(
        model=factory(
            [
                (
                    "Test response",
                    [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'custom config works'"}],
                )
            ]
        ),
        env=LocalEnvironment(),
        **{
            **config,
            "system_template": "You are a test assistant.",
            "instance_template": "Task: {{task}}. Return bash command.",
            "step_limit": 2,
            "cost_limit": 1.0,
        },
    )

    info = agent.run("Test custom config")
    assert info["exit_status"] == "Submitted"
    assert info["submission"] == "custom config works\n"
    assert get_text(agent.messages[0]) == "You are a test assistant."
    assert "Test custom config" in get_text(agent.messages[1])


def test_render_template_model_stats(model_factory):
    """Test that render_template has access to n_model_calls and model_cost from agent."""
    factory, config = model_factory
    agent = DefaultAgent(
        model=factory(
            [
                ("Test 1", [{"command": "echo 'test1'"}]),
                ("Test 2", [{"command": "echo 'test2'"}]),
            ]
        ),
        env=LocalEnvironment(),
        **config,
    )

    # Make some calls through the agent to generate stats
    agent.add_messages({"role": "system", "content": "test"}, {"role": "user", "content": "test"})
    agent.query()
    agent.query()

    # Test template rendering with agent stats
    template = "Calls: {{n_model_calls}}, Cost: {{model_cost}}"
    assert agent._render_template(template) == "Calls: 2, Cost: 2.0"


def test_messages_include_timestamps(model_factory):
    """Test that assistant and observation messages include timestamps."""
    factory, config = model_factory
    agent = DefaultAgent(
        model=factory(
            [
                ("Response 1", [{"command": "echo 'test1'"}]),
                ("Response 2", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'"}]),
            ]
        ),
        env=LocalEnvironment(),
        **config,
    )

    agent.run("Test timestamps")

    # Assistant messages should have timestamps
    assistant_msgs = [msg for msg in agent.messages if is_assistant_message(msg)]
    assert all("timestamp" in msg.get("extra", {}) for msg in assistant_msgs)
    # Timestamps should be numeric (floats from time.time())
    all_timestamped = [msg for msg in agent.messages if "timestamp" in msg.get("extra", {})]
    assert all(isinstance(msg["extra"]["timestamp"], float) for msg in all_timestamped)


def test_message_history_tracking(model_factory):
    """Test that messages are properly added and tracked."""
    factory, config = model_factory
    agent = DefaultAgent(
        model=factory(
            [
                ("Response 1", [{"command": "echo 'test1'"}]),
                ("Response 2", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'"}]),
            ]
        ),
        env=LocalEnvironment(),
        **config,
    )

    info = agent.run("Track messages")
    assert info["exit_status"] == "Submitted"
    assert info["submission"] == "done\n"

    # Should have 6 messages: system, user, assistant, observation, assistant, exit
    assert len(agent.messages) == 6
    # First two are system and user
    assert get_text(agent.messages[0])  # system has content
    assert get_text(agent.messages[1])  # user has content
    # Third is assistant response
    assert is_assistant_message(agent.messages[2])
    # Fourth is observation
    assert is_observation_message(agent.messages[3])
    # Fifth is assistant response
    assert is_assistant_message(agent.messages[4])


def test_step_adds_messages(model_factory):
    """Test that step adds assistant and observation messages."""
    factory, config = model_factory
    agent = DefaultAgent(
        model=factory([("Test command", [{"command": "echo 'hello'"}])]),
        env=LocalEnvironment(),
        **config,
    )

    agent.add_messages({"role": "system", "content": "system message"})
    agent.add_messages({"role": "user", "content": "user message"})

    initial_count = len(agent.messages)
    agent.step()

    # step() should add assistant message + observation message
    assert len(agent.messages) == initial_count + 2
    assert is_assistant_message(agent.messages[-2])
    assert agent.messages[-2]["extra"]["actions"][0]["command"] == "echo 'hello'"
    assert is_observation_message(agent.messages[-1])
    assert "returncode" in get_observation_text(agent.messages[-1])


def test_observations_captured(model_factory):
    """Test intermediate outputs are captured correctly."""
    factory, config = model_factory
    agent = DefaultAgent(
        model=factory(
            [
                ("Step 1", [{"command": "echo 'first'"}]),
                ("Step 2", [{"command": "echo 'second'"}]),
                ("Final", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'"}]),
            ]
        ),
        env=LocalEnvironment(),
        **{**config, "cost_limit": 5.0},
    )

    agent.run("Multi-step task")
    observations = [get_observation_text(msg) for msg in agent.messages if is_observation_message(msg)]
    assert len(observations) == 2
    assert "first" in observations[0]
    assert "second" in observations[1]


def test_wall_time_limit_enforcement(model_factory):
    """Test agent stops when wall-clock time limit is reached."""
    factory, config = model_factory
    agent = DefaultAgent(
        model=factory(
            [
                ("Slow command", [{"command": "sleep 2"}]),
                ("Should not run", [{"command": "echo 'unreachable'"}]),
            ]
        ),
        env=LocalEnvironment(),
        **{**config, "wall_time_limit_seconds": 1},
    )

    info = agent.run("Test wall time limit")
    assert info["exit_status"] == "TimeExceeded"
    assert agent.n_calls == 1


def test_wall_time_limit_template_vars(model_factory):
    """Test that elapsed_seconds and wall_time_limit_seconds are available as template vars."""
    factory, config = model_factory
    agent = DefaultAgent(
        model=factory([("Test", [{"command": "echo 'test'"}])]),
        env=LocalEnvironment(),
        **{**config, "wall_time_limit_seconds": 3600},
    )
    agent.add_messages({"role": "system", "content": "test"}, {"role": "user", "content": "test"})
    tvars = agent.get_template_vars()
    assert isinstance(tvars["elapsed_seconds"], int)
    assert tvars["wall_time_limit_seconds"] == 3600


def test_empty_actions_handling(model_factory):
    """Test agent handles empty actions (continues without error)."""
    factory, config = model_factory
    agent = DefaultAgent(
        model=factory(
            [
                ("No actions here", []),  # Empty actions list
                ("Now with action", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'"}]),
            ]
        ),
        env=LocalEnvironment(),
        **config,
    )

    info = agent.run("Test empty actions")
    assert info["exit_status"] == "Submitted"
    assert info["submission"] == "done\n"
    assert agent.n_calls == 2


class _FlakyToolcallModel(DeterministicToolcallModel):
    """Like DeterministicToolcallModel, but raises FormatError (as the real LitellmModel now does
    on a truncated / no-tool-call turn) for any output marked {"_format_error": True}."""

    def query(self, messages, **kwargs):
        self.current_index += 1
        output = self.config.outputs[self.current_index]
        if output.get("_format_error"):
            raise FormatError(
                {
                    "role": "user",
                    "content": "No tool calls found in the response.",
                    "extra": {"interrupt_type": "FormatError"},
                }
            )
        return output


def test_repeated_format_errors_terminate_cleanly(toolcall_config):
    """With max_consecutive_format_errors set, a run that keeps producing no-tool-call / truncation
    turns stops cleanly with exit_status=RepeatedFormatError instead of looping until the budget is
    gone."""
    outputs = [{"_format_error": True} for _ in range(5)]
    agent = DefaultAgent(
        model=_FlakyToolcallModel(outputs=outputs),
        env=LocalEnvironment(),
        **{**toolcall_config, "max_consecutive_format_errors": 2},
    )
    info = agent.run("Test repeated format errors")
    assert info["exit_status"] == "RepeatedFormatError"
    assert agent.n_calls == 2  # stopped at the 2nd consecutive error, didn't burn all 5


def test_format_error_counter_resets_on_success(toolcall_config):
    """A successful tool call between format errors resets the consecutive counter, so isolated
    errors don't accumulate to the termination threshold."""
    good = make_tc_model([("listing", [{"command": "echo hello"}])]).config.outputs[0]
    submit = make_tc_model(
        [("done", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho ok"}])]
    ).config.outputs[0]
    # error, success (reset), error, submit -> never 2 in a row, so it must NOT terminate early.
    outputs = [{"_format_error": True}, good, {"_format_error": True}, submit]
    agent = DefaultAgent(
        model=_FlakyToolcallModel(outputs=outputs),
        env=LocalEnvironment(),
        **{**toolcall_config, "max_consecutive_format_errors": 2},
    )
    info = agent.run("Test counter reset")
    assert info["exit_status"] == "Submitted"
    assert info["submission"] == "ok\n"


class _BilledFormatErrorModel(DeterministicToolcallModel):
    """Bills the call and then fails to parse it, the way the real model classes do: the cost is
    charged to the global stats and persisted on the FormatError before it propagates."""

    def query(self, messages, **kwargs):
        GLOBAL_MODEL_STATS.add(self.config.cost_per_call)
        raise FormatError(
            {
                "role": "user",
                "content": "No tool calls found in the response.",
                "extra": {"interrupt_type": "FormatError", "cost": self.config.cost_per_call},
            }
        )


def test_format_errors_count_against_cost_limit(toolcall_config, reset_global_stats):
    """Turns that fail to parse are still billed, so they have to count against cost_limit.
    step_limit is only a backstop here: if the format-error path stopped charging, the run would
    run on to that limit with agent.cost still at zero."""
    agent = DefaultAgent(
        model=_BilledFormatErrorModel(outputs=[], cost_per_call=1.0),
        env=LocalEnvironment(),
        **{**toolcall_config, "cost_limit": 2.5, "step_limit": 8, "max_consecutive_format_errors": 0},
    )

    info = agent.run("Test billed format errors")
    assert info["exit_status"] == "LimitsExceeded"
    assert agent.n_calls == 3
    assert agent.cost == 3.0
    assert agent.cost == GLOBAL_MODEL_STATS.cost


def test_context_compaction_triggers_with_tiny_context_limit(monkeypatch):
    """Tiny MAX_INPUT_TOKENS provides a manual path to force compaction."""
    monkeypatch.setenv("MAX_INPUT_TOKENS", "400")
    monkeypatch.setenv("MSWEA_CONTEXT_COMPACT_AT", "19")
    monkeypatch.setenv("MSWEA_CONTEXT_COMPACT_TO", "50")
    monkeypatch.setenv("MSWEA_CONTEXT_TAIL_TARGET_PERCENT", "25")
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                make_output("step 1", [{"command": "echo one"}]),
                make_output("step 2", [{"command": "echo two"}]),
                make_output("step 3", [{"command": "echo three"}]),
                make_output(
                    "current objective: finish\n"
                    "user constraints: preserve behavior\n"
                    "files inspected: none\n"
                    "files modified: none\n"
                    "commands run: echo one, echo two, echo three\n"
                    "test results: not run\n"
                    "failed approaches: none\n"
                    "current plan: submit\n"
                    "remaining TODOs: submit final output\n"
                    "important facts that must not be forgotten: prior echo commands completed",
                    [],
                ),
                make_output(
                    "done",
                    [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'"}],
                ),
            ]
        ),
        env=LocalEnvironment(),
        **minimal_agent_config(),
    )

    info = agent.run("Force compaction")

    assert info["exit_status"] == "Submitted"
    summaries = [msg for msg in agent.messages if msg.get("extra", {}).get("compact_summary")]
    assert len(summaries) == 1
    assert "current objective" in get_text(summaries[0])
    assert get_text(agent.messages[0])
    assert "Force compaction" in get_text(agent.messages[1])


def test_context_compaction_skips_below_threshold(default_config, monkeypatch):
    monkeypatch.setenv("MAX_INPUT_TOKENS", "100000")
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                make_output("step 1", [{"command": "echo one"}]),
                make_output(
                    "done",
                    [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'"}],
                ),
            ]
        ),
        env=LocalEnvironment(),
        **{**default_config, "cost_limit": 10.0},
    )

    info = agent.run("Do not compact")

    assert info["exit_status"] == "Submitted"
    assert [msg for msg in agent.messages if msg.get("extra", {}).get("compact_summary")] == []
    assert len(agent.messages) == 6


def test_context_compaction_uses_env_thresholds(monkeypatch):
    monkeypatch.setenv("MAX_INPUT_TOKENS", "600")
    monkeypatch.setenv("MSWEA_CONTEXT_COMPACT_AT", "20")
    monkeypatch.setenv("MSWEA_CONTEXT_COMPACT_TO", "50")
    monkeypatch.setenv("MSWEA_CONTEXT_TAIL_TARGET_PERCENT", "25")
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                make_output("step 1", [{"command": "echo one"}]),
                make_output("step 2", [{"command": "echo two"}]),
                make_output("step 3", [{"command": "echo three"}]),
                make_output("step 4", [{"command": "echo four"}]),
                make_output(
                    "summary with current objective and remaining TODOs",
                    [],
                ),
                make_output(
                    "done",
                    [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'"}],
                ),
            ]
        ),
        env=LocalEnvironment(),
        **minimal_agent_config(),
    )

    info = agent.run("Use env thresholds")

    assert info["exit_status"] == "Submitted"
    assert [msg for msg in agent.messages if msg.get("extra", {}).get("compact_summary")]


def test_debug_exchange_log_records_model_and_action_events(default_config, tmp_path, monkeypatch):
    monkeypatch.setenv("MAX_INPUT_TOKENS", "100000")
    debug_path = tmp_path / "debug-exchanges.jsonl"
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                make_output("step 1", [{"command": "echo one"}]),
                make_output(
                    "done",
                    [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'"}],
                ),
            ]
        ),
        env=LocalEnvironment(),
        **{**default_config, "cost_limit": 10.0, "debug_exchange_path": debug_path},
    )

    info = agent.run("Debug normal run")
    events = [json.loads(line) for line in debug_path.read_text().splitlines()]

    assert info["exit_status"] == "Submitted"
    assert [event for event in events if event["event"] == "run_start"]
    assert len([event for event in events if event["event"] == "model_call"]) == 2
    assert [event for event in events if event["event"] == "action_execution"]
    assert [event for event in events if event["event"] == "run_end"]
    assert not [event for event in events if event["event"] == "compaction_triggered"]


def test_debug_exchange_log_records_provider_request_and_response(default_config, tmp_path, monkeypatch):
    monkeypatch.setenv("MAX_INPUT_TOKENS", "100000")
    debug_path = tmp_path / "debug-exchanges.jsonl"
    agent = DefaultAgent(
        model=DebugProviderDeterministicModel(
            outputs=[
                make_output("step 1", [{"command": "echo one"}]),
                make_output("done", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'"}]),
            ]
        ),
        env=LocalEnvironment(),
        **{**default_config, "cost_limit": 10.0, "debug_exchange_path": debug_path},
    )

    agent.run("Debug provider exchange")
    events = [json.loads(line) for line in debug_path.read_text().splitlines()]
    model_event = [event for event in events if event["event"] == "model_call"][0]

    assert model_event["provider_request"]["provider"] == "debug-test"
    assert model_event["provider_request"]["messages"][0]["role"] == "system"
    assert model_event["provider_response"]["message"]["content"] == "step 1"


def test_readable_debug_exchange_writes_request_before_response(default_config, tmp_path, monkeypatch):
    monkeypatch.setenv("MAX_INPUT_TOKENS", "100000")
    readable_path = tmp_path / "debug-exchanges-readable.md"
    agent = DefaultAgent(
        model=AssertingReadableRequestWrittenModel(
            readable_path=readable_path,
            outputs=[make_output("done", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'"}])],
        ),
        env=LocalEnvironment(),
        **{**default_config, "cost_limit": 10.0, "debug_exchange_readable_path": readable_path},
    )

    agent.run("Debug readable request timing")
    text = readable_path.read_text()

    assert "## Exchange 0: `model_call`" in text
    assert "### Sent to provider" in text
    assert "### Provider response" in text
    assert text.index("### Sent to provider") < text.index("### Provider response")
    assert "done" in text


def test_readable_debug_exchange_requires_exact_model_bound_messages(default_config, tmp_path, monkeypatch):
    monkeypatch.setenv("MAX_INPUT_TOKENS", "100000")
    agent = DefaultAgent(
        model=DeterministicModel(outputs=[make_output("done", [])]),
        env=LocalEnvironment(),
        **{**default_config, "cost_limit": 10.0, "debug_exchange_readable_path": tmp_path / "readable.md"},
    )

    with pytest.raises(RuntimeError, match="report the exact provider request payload"):
        agent.run("Debug readable strict mode")


def test_readable_debug_exchange_clips_tool_results_only_in_markdown(default_config, tmp_path, monkeypatch):
    monkeypatch.setenv("MAX_INPUT_TOKENS", "100000")
    readable_path = tmp_path / "debug-exchanges-readable.md"
    command = "printf '" + "\\n".join(f"line {i}" for i in range(1, 31)) + "\\n'"
    agent = DefaultAgent(
        model=PreparedDeterministicModel(
            outputs=[
                make_output("step 1", [{"command": command}]),
                make_output("done", [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'"}]),
            ]
        ),
        env=LocalEnvironment(),
        **{**default_config, "cost_limit": 10.0, "debug_exchange_readable_path": readable_path},
    )

    agent.run("Debug readable clipping")
    text = readable_path.read_text()

    assert "line 25" in text
    assert "...[clipped after 25 lines; " in text
    clipped_observation = text.split("<output>", 1)[1].split("</output>", 1)[0]
    assert "line 26" not in clipped_observation
    assert "\\nline 2" not in clipped_observation
    assert any("line 30" in msg.get("extra", {}).get("raw_output", "") for msg in agent.messages)


def test_debug_exchange_log_records_ollama_provider_payload(tmp_path, monkeypatch):
    monkeypatch.setenv("MAX_INPUT_TOKENS", "100000")
    debug_path = tmp_path / "debug-exchanges.jsonl"
    response = type("Response", (), {})()
    response.raise_for_status = lambda: None
    response.json = lambda: {
        "message": {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"function": {"name": "bash", "arguments": {"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"}}}
            ],
        }
    }
    response.iter_lines = lambda: [json.dumps(response.json()).encode()]

    def fake_post(*args, **kwargs):
        return response

    monkeypatch.setattr("requests.post", fake_post)
    agent = DefaultAgent(
        model=OllamaModel(model_name="qwen3-coder:30b"),
        env=LocalEnvironment(),
        **minimal_agent_config(debug_exchange_path=debug_path),
    )

    agent.run("Debug Ollama provider payload")
    events = [json.loads(line) for line in debug_path.read_text().splitlines()]
    model_event = [event for event in events if event["event"] == "model_call"][0]

    assert model_event["provider_request"]["provider"] == "ollama"
    assert model_event["provider_request"]["payload"]["tools"][0]["function"]["name"] == "bash"
    assert model_event["provider_request"]["body"] == json.dumps(model_event["provider_request"]["payload"])
    assert model_event["provider_response"]["message"]["tool_calls"][0]["function"]["name"] == "bash"


def test_readable_debug_exchange_records_ollama_provider_payload(tmp_path, monkeypatch):
    monkeypatch.setenv("MAX_INPUT_TOKENS", "100000")
    readable_path = tmp_path / "debug-exchanges-readable.md"
    response = type("Response", (), {})()
    response.raise_for_status = lambda: None
    response.json = lambda: {
        "message": {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"function": {"name": "bash", "arguments": {"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"}}}
            ],
        }
    }
    response.iter_lines = lambda: [json.dumps(response.json()).encode()]

    monkeypatch.setattr("requests.post", lambda *args, **kwargs: response)
    agent = DefaultAgent(
        model=OllamaModel(model_name="qwen3-coder:30b"),
        env=LocalEnvironment(),
        **minimal_agent_config(debug_exchange_readable_path=readable_path),
    )

    agent.run("Debug readable Ollama provider payload")
    text = readable_path.read_text()

    assert "### Sent to provider" in text
    assert "`tools`" in text
    assert "`model`" in text
    assert "`messages`" in text
    assert "`stream`" in text
    assert "`options`" in text
    assert "### Provider response" in text


def test_debug_exchange_log_records_compaction_summary_exchange(tmp_path, monkeypatch):
    monkeypatch.setenv("MAX_INPUT_TOKENS", "600")
    monkeypatch.setenv("MSWEA_CONTEXT_COMPACT_AT", "19")
    monkeypatch.setenv("MSWEA_CONTEXT_COMPACT_TO", "50")
    monkeypatch.setenv("MSWEA_CONTEXT_TAIL_TARGET_PERCENT", "25")
    debug_path = tmp_path / "debug-exchanges.jsonl"
    agent = DefaultAgent(
        model=DeterministicModel(
            outputs=[
                make_output("step 1", [{"command": "echo one"}]),
                make_output("step 2", [{"command": "echo two"}]),
                make_output("step 3", [{"command": "echo three"}]),
                make_output("summary with current objective and remaining TODOs", []),
                make_output(
                    "done",
                    [{"command": "echo 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT'\necho 'done'"}],
                ),
            ]
        ),
        env=LocalEnvironment(),
        **minimal_agent_config(debug_exchange_path=debug_path),
    )

    info = agent.run("Debug compaction")
    events = [json.loads(line) for line in debug_path.read_text().splitlines()]
    summary_events = [event for event in events if event["event"] == "compaction_summary_call"]

    assert info["exit_status"] == "Submitted"
    triggered = [event for event in events if event["event"] == "compaction_triggered"]
    assert triggered
    assert triggered[0]["target"] == 300
    assert triggered[0]["tail_target"] == 75
    assert triggered[0]["summary_budget"] == 300 - triggered[0]["head_tokens"] - triggered[0]["tail_tokens"]
    assert triggered[0]["tail_message_count"] > 0
    assert summary_events
    assert "Messages to summarize" in summary_events[0]["request_messages"][1]["content"]
    assert f"Target length: {triggered[0]['summary_budget']} tokens." in summary_events[0]["request_messages"][1]["content"]
    assert "Use most of this budget for a detailed summary" in summary_events[0]["request_messages"][1]["content"]
    assert summary_events[0]["response_message"]["content"] == "summary with current objective and remaining TODOs"
    assert summary_events[0]["summary"] == "summary with current objective and remaining TODOs"
    assert [event for event in events if event["event"] == "compaction_finished"]


def test_context_compaction_errors_when_newest_tail_group_does_not_fit(monkeypatch):
    monkeypatch.setenv("MAX_INPUT_TOKENS", "400")
    monkeypatch.setenv("MSWEA_CONTEXT_TAIL_TARGET_PERCENT", "10")
    agent = DefaultAgent(
        model=DeterministicModel(outputs=[]),
        env=LocalEnvironment(),
        **minimal_agent_config(),
    )
    agent.messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "old"},
        {"role": "user", "content": "old observation"},
        {
            "role": "assistant",
            "content": "newest action",
            "extra": {"actions": [{"command": "echo newest"}]},
        },
        {"role": "user", "content": "returncode " + ("x" * 500)},
    ]

    with pytest.raises(RuntimeError, match="Newest tail message group does not fit"):
        agent._choose_tail_start(agent.messages, token_budget=10)


def test_context_compaction_tail_keeps_action_result_pair_whole(monkeypatch):
    monkeypatch.setenv("MAX_INPUT_TOKENS", "800")
    agent = DefaultAgent(
        model=DeterministicModel(outputs=[]),
        env=LocalEnvironment(),
        **minimal_agent_config(),
    )
    agent.messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "old"},
        {"role": "user", "content": "old observation"},
        {
            "role": "assistant",
            "content": "newest action",
            "extra": {"actions": [{"command": "echo newest"}]},
        },
        {"role": "user", "content": "returncode newest"},
    ]

    tail_start = agent._choose_tail_start(agent.messages, token_budget=30)

    assert tail_start == 4
    assert agent.messages[tail_start:][0]["role"] == "assistant"
    assert agent.messages[tail_start:][1]["role"] == "user"


def test_context_compaction_estimates_prepared_messages_without_extra(monkeypatch):
    monkeypatch.setenv("MAX_INPUT_TOKENS", "1000")
    monkeypatch.setenv("MSWEA_CONTEXT_COMPACT_AT", "60")
    agent = DefaultAgent(
        model=DeterministicModel(outputs=[]),
        env=LocalEnvironment(),
        **minimal_agent_config(),
    )
    agent.messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "content": "small",
            "extra": {"response": {"raw": "x" * 10000}},
        },
        {
            "role": "user",
            "content": "small observation",
            "extra": {"raw_output": "y" * 10000},
        },
    ]

    assert agent._estimate_tokens(agent.messages) > 600
    assert agent._estimate_api_tokens(agent.messages) < 600


def test_context_compaction_chunks_middle_before_oversized_summary_call(tmp_path):
    class RecordingTextModel(DeterministicModel):
        def __init__(self):
            super().__init__(outputs=[])
            self.text_calls = []

        def query_text(self, messages, **kwargs):
            self.text_calls.append((messages, kwargs))
            return {"choices": [{"message": {"role": "assistant", "content": f"summary {len(self.text_calls)}"}}]}

        def compaction_query_kwargs(self):
            return {"think": False}

    model = RecordingTextModel()
    debug_path = tmp_path / "debug.jsonl"
    agent = DefaultAgent(
        model=model,
        env=LocalEnvironment(),
        **minimal_agent_config(debug_exchange_path=debug_path),
    )
    middle = [{"role": "user", "content": f"message {i} " + ("x" * 900)} for i in range(8)]

    summary = agent._summarize_bounded(middle, token_budget=100, context_limit=1000)
    events = [json.loads(line) for line in debug_path.read_text().splitlines()]

    assert summary.startswith("summary")
    assert len(model.text_calls) > 1
    for messages, kwargs in model.text_calls:
        assert agent._estimate_api_tokens(messages) + kwargs["max_tokens"] <= 1000
        assert kwargs["think"] is False
    assert [event for event in events if event["event"] == "compaction_chunk_planned"]
    assert [event for event in events if event["event"] == "compaction_chunk_summary_call"]
    assert [event for event in events if event["event"] == "compaction_final_summary_call"]
