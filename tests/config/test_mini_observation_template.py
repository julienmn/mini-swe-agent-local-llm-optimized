from dataclasses import dataclass
from pathlib import Path

import yaml
from jinja2 import StrictUndefined, Template


@dataclass
class MockOutput:
    returncode: int
    output: str
    exception_info: str = ""


def _render_mini_observation(output: MockOutput, **template_vars) -> str:
    config_path = Path(__file__).parent.parent.parent / "src" / "minisweagent" / "config" / "mini.yaml"
    config = yaml.safe_load(config_path.read_text())
    template = Template(config["model"]["observation_template"], undefined=StrictUndefined)
    return template.render(output=output, **template_vars)


def test_mini_observation_template_uses_existing_default_limit():
    output = MockOutput(returncode=0, output="A" * 6000 + "B" * 5000)

    result = _render_mini_observation(output)

    assert '"output_head": "' + ("A" * 5000) in result
    assert '"output_tail": "' + ("B" * 5000) in result
    assert '"elided_chars": 1000' in result


def test_mini_observation_template_uses_configured_limit_split_half_head_tail():
    output = MockOutput(returncode=0, output="A" * 3000 + "B" * 3000)

    result = _render_mini_observation(output, MSWEA_OBSERVATION_OUTPUT_LIMIT="5000")

    assert '"output_head": "' + ("A" * 2500) in result
    assert '"output_tail": "' + ("B" * 2500) in result
    assert '"elided_chars": 1000' in result


def test_mini_observation_template_short_output_uses_full_output_with_configured_limit():
    output = MockOutput(returncode=0, output="short output")

    result = _render_mini_observation(output, MSWEA_OBSERVATION_OUTPUT_LIMIT="5000")

    assert '"output": "short output"' in result
    assert "output_head" not in result
    assert "output_tail" not in result
