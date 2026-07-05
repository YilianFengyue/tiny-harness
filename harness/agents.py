"""Built-in sub-agent definitions and tool filtering for CH09."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TYPE_CHECKING

from .coordinator import WORKER_PROMPT

if TYPE_CHECKING:
    from .tools.registry import ToolSpec

AgentSource = Literal["built-in"]


@dataclass(frozen=True)
class AgentDefinition:
    agent_type: str
    description: str
    system_prompt: str
    tools: tuple[str, ...] | None = None
    disallowed_tools: tuple[str, ...] = ()
    model: str | None = None
    max_turns: int | None = None
    background: bool = False
    color: str | None = None
    require_read_only_tools: bool = False
    source: AgentSource = "built-in"
    metadata: dict[str, object] = field(default_factory=dict)


EXPLORE_PROMPT = """You are an Explore sub-agent.
Your job is read-only investigation: inspect files, search symbols, and report
facts with file paths and line numbers. Do not modify files. Do not ask the
user questions. Keep the report concise and actionable."""

PLAN_PROMPT = """You are a Plan sub-agent.
Use read-only investigation to produce a concrete implementation plan. End with
key files and risks. Do not modify files. Do not ask the user questions."""

GENERAL_PROMPT = """You are a General sub-agent.
Complete the delegated coding task autonomously within the provided scope.
Use tools to inspect and verify. Report what changed, tests run, and any risks."""

VERIFY_PROMPT = """You are a Verification sub-agent.
Act adversarially: try to prove the implementation is broken. Run relevant
tests or checks when possible. Do not modify project files. Report failures,
edge cases, and missing verification."""


READ_ONLY_TOOLS = (
    "read_file",
    "glob_files",
    "grep",
    "file_info",
    "show_diff",
    "list_files",
    "bash",
)


BUILT_IN_AGENTS: tuple[AgentDefinition, ...] = (
    AgentDefinition(
        "explore",
        "Read-only code explorer for locating files, symbols, and causes.",
        EXPLORE_PROMPT,
        tools=READ_ONLY_TOOLS,
        max_turns=10,
        color="blue",
        require_read_only_tools=True,
    ),
    AgentDefinition(
        "plan",
        "Read-only planning agent for implementation strategy and risk analysis.",
        PLAN_PROMPT,
        tools=READ_ONLY_TOOLS,
        max_turns=10,
        color="green",
        require_read_only_tools=True,
    ),
    AgentDefinition(
        "general",
        "General implementation agent for scoped coding tasks.",
        GENERAL_PROMPT,
        disallowed_tools=("agent",),
        max_turns=20,
        color="yellow",
    ),
    AgentDefinition(
        "verify",
        "Read-only adversarial verification agent for tests and edge cases.",
        VERIFY_PROMPT,
        tools=READ_ONLY_TOOLS,
        max_turns=12,
        color="red",
        require_read_only_tools=True,
    ),
    AgentDefinition(
        "worker",
        "Coordinator worker for research, implementation, and verification.",
        WORKER_PROMPT,
        disallowed_tools=("agent",),
        max_turns=20,
        background=True,
        color="magenta",
    ),
)


def get_builtin_agents() -> tuple[AgentDefinition, ...]:
    return BUILT_IN_AGENTS


def get_agent_definitions(workdir: Path | None = None) -> tuple[AgentDefinition, ...]:
    agents = list(BUILT_IN_AGENTS)
    if workdir is not None:
        agents.extend(load_project_agents(workdir))
    merged: dict[str, AgentDefinition] = {}
    for agent in agents:
        merged[agent.agent_type] = agent
    return tuple(merged.values())


def get_agent_definition(agent_type: str | None,
                         workdir: Path | None = None) -> AgentDefinition | None:
    target = agent_type or "general"
    return next((agent for agent in get_agent_definitions(workdir)
                 if agent.agent_type == target), None)


def agent_tool_names(agent: AgentDefinition) -> set[str]:
    from .tools.registry import REGISTRY

    if agent.tools is None:
        names = set(REGISTRY)
    else:
        names = set(agent.tools)
    names.difference_update(agent.disallowed_tools)
    names.discard("agent")
    return {name for name in names if name in REGISTRY}


def agent_tool_schemas(agent: AgentDefinition) -> list[dict]:
    from .tools.registry import REGISTRY

    names = agent_tool_names(agent)
    specs = [REGISTRY[name] for name in sorted(names)]
    return [schema_for_tool(spec) for spec in specs]


def schema_for_tool(spec: "ToolSpec") -> dict:
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.parameters,
            "strict": True,
        },
    }


def load_project_agents(workdir: Path) -> tuple[AgentDefinition, ...]:
    agents_dir = Path(workdir) / ".tiny-harness" / "agents"
    if not agents_dir.exists():
        return ()
    agents: list[AgentDefinition] = []
    for path in sorted(agents_dir.glob("*.md")):
        agent = parse_agent_markdown(path)
        if agent:
            agents.append(agent)
    return tuple(agents)


def parse_agent_markdown(path: Path) -> AgentDefinition | None:
    raw = path.read_text(encoding="utf-8-sig")
    if not raw.startswith("---"):
        return None
    _, _, rest = raw.partition("---")
    frontmatter, sep, body = rest.partition("---")
    if not sep:
        return None
    data = _parse_frontmatter(frontmatter)
    name = str(data.get("name") or "").strip()
    description = str(data.get("description") or "").strip()
    prompt = body.strip()
    if not name or not description or not prompt:
        return None
    return AgentDefinition(
        agent_type=name,
        description=description,
        system_prompt=prompt,
        tools=_as_tuple(data.get("tools")),
        disallowed_tools=_as_tuple(data.get("disallowedTools")) or (),
        model=str(data["model"]).strip() if data.get("model") else None,
        max_turns=int(data["maxTurns"]) if data.get("maxTurns") else None,
        background=_as_bool(data.get("background")),
        color=str(data["color"]).strip() if data.get("color") else None,
        require_read_only_tools=_as_bool(data.get("readOnly")),
    )


def _parse_frontmatter(text: str) -> dict[str, object]:
    data: dict[str, object] = {}
    current_key: str | None = None
    current_items: list[str] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        if not line.strip():
            continue
        stripped = line.strip()
        if stripped.startswith("- ") and current_key:
            current_items.append(stripped[2:].strip().strip("'\""))
            data[current_key] = list(current_items)
            continue
        current_key = None
        current_items = []
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip()
        value = value.strip()
        if value == "":
            current_key = key
            current_items = []
            data[key] = current_items
        else:
            data[key] = _parse_frontmatter_value(value)
    return data


def _parse_frontmatter_value(value: str) -> object:
    value = value.strip().strip("'\"")
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [item.strip().strip("'\"") for item in inner.split(",") if item.strip()]
    lower = value.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    if value.isdigit():
        return int(value)
    return value


def _as_tuple(value: object) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return tuple(str(item).strip() for item in value if str(item).strip())
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    return None


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def format_agent_listing(workdir: Path | None = None) -> str:
    lines = []
    for agent in get_agent_definitions(workdir):
        tools = "all" if agent.tools is None else ", ".join(agent.tools)
        if agent.disallowed_tools:
            tools += f" except {', '.join(agent.disallowed_tools)}"
        lines.append(f"- {agent.agent_type}: {agent.description} (tools: {tools})")
    return "\n".join(lines)
