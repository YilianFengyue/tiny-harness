"""工具注册表：@tool 装饰器、strict schema 导出、统一执行与截断。

设计要点（DESIGN.md §Tools）：
- 工具描述按"给新员工写入职文档"的标准写——Anthropic 在 SWE-bench 上仅靠
  精修工具描述就显著降低错误率，这是 harness 里性价比最高的优化点。
- 全部启用 strict: true（additionalProperties=false + 字段全 required，
  可选语义用 ["T","null"] 表达），让 API 层面保证参数合法，消灭一类参数幻觉。
- 工具结果统一截断（默认 20K 字符）：一条 cat 大文件就能炸掉上下文，
  截断标记会提示模型改用分页/bash 管道。
- 错误信息必须可操作：告诉模型"接下来该怎么办"，而不是裸 traceback。
"""
from __future__ import annotations

import copy
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from ..cancel import CancellationToken, CancelledError

ProgressCallback = Callable[[dict], None]
ContextModifier = Callable[["ToolRuntimeState"], dict | None]
ToolValidator = Callable[["ToolContext", dict], None]
PermissionResolver = Callable[[str, dict, object, object, "ToolContext"], str]


@dataclass
class ToolRuntimeState:
    """Mutable state shared by tool calls in one agent run.

    This is the tiny-harness equivalent of Claude Code's ToolUseContext
    mutation channel. Tools do not rewrite the message history directly; they
    return a context modifier, and the loop applies it at a deterministic point.
    """
    read_files: dict[str, dict] = field(default_factory=dict)
    file_history: list[dict] = field(default_factory=list)
    persisted_results: dict[str, str] = field(default_factory=dict)
    permission_context: object | None = None
    permission_resolver: PermissionResolver | None = None
    config: object | None = None
    provider: object | None = None
    agent_id: str | None = None
    agent_type: str | None = None
    agent_depth: int = 0
    allowed_tools: set[str] | None = None
    disallowed_tools: set[str] = field(default_factory=set)
    require_read_only_tools: bool = False
    event_callback: ProgressCallback | None = None
    messages: list[dict] | None = None
    background_agents: object | None = None


@dataclass
class ToolContext:
    """工具执行环境（loop 注入）。"""
    workdir: Path
    bash_timeout: int = 60
    output_limit: int = 20_000
    cancel_token: CancellationToken | None = None
    progress_callback: ProgressCallback | None = None
    runtime: ToolRuntimeState = field(default_factory=ToolRuntimeState)

    def throw_if_cancelled(self) -> None:
        if self.cancel_token:
            self.cancel_token.throw_if_cancelled()

    def progress(self, **payload) -> None:
        if self.progress_callback:
            self.progress_callback(payload)


@dataclass
class ToolResult:
    text: str
    ok: bool = True
    truncated: bool = False
    duration_ms: int = 0
    persisted_path: str | None = None
    context_modifier: ContextModifier | None = None
    context_modified: dict | None = None
    error_kind: str | None = None


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict
    fn: Callable[..., str | ToolResult]
    dangerous_check: Callable[[dict], str | None] | None = None  # 返回命中原因或 None
    aliases: tuple[str, ...] = ()
    read_only: bool | Callable[[dict], bool] = False
    concurrency_safe: bool | Callable[[dict], bool] = False
    destructive: bool | Callable[[dict], bool] = False
    max_result_size_chars: int | None = None
    validate_input: ToolValidator | None = None


REGISTRY: dict[str, ToolSpec] = {}


def tool(name: str, description: str, parameters: dict,
         dangerous_check: Callable[[dict], str | None] | None = None,
         aliases: tuple[str, ...] | list[str] = (),
         read_only: bool | Callable[[dict], bool] = False,
         concurrency_safe: bool | Callable[[dict], bool] = False,
         destructive: bool | Callable[[dict], bool] = False,
         max_result_size_chars: int | None = None,
         validate_input: ToolValidator | None = None):
    """注册工具。fn 签名为 fn(ctx: ToolContext, **arguments) -> str，
    抛 ToolError 表示可恢复的业务错误（回传模型自纠），其他异常视为 bug。"""
    def deco(fn):
        REGISTRY[name] = ToolSpec(
            name=name,
            description=description,
            parameters=_strictify(parameters),
            fn=fn,
            dangerous_check=dangerous_check,
            aliases=tuple(aliases),
            read_only=read_only,
            concurrency_safe=concurrency_safe,
            destructive=destructive,
            max_result_size_chars=max_result_size_chars,
            validate_input=validate_input,
        )
        return fn
    return deco


class ToolError(Exception):
    """可恢复的工具错误：消息会以 is_error 语义回传给模型。"""


def _strictify(params: dict) -> dict:
    """补全 strict 模式要求：每层 object 都 additionalProperties=false 且字段全 required。"""
    if params.get("type") == "object":
        params.setdefault("properties", {})
        params["required"] = list(params["properties"].keys())
        params["additionalProperties"] = False
        for sub in params["properties"].values():
            _strictify(sub)
    elif params.get("type") == "array" and "items" in params:
        _strictify(params["items"])
    return params


def openai_tool_schemas(workdir: Path | None = None,
                        coordinator_mode: bool = False) -> list[dict]:
    return [
        {"type": "function",
         "function": {"name": s.name, "description": _schema_description(s, workdir, coordinator_mode),
                      "parameters": _schema_parameters(s, coordinator_mode),
                      "strict": True}}
        for s in sorted(REGISTRY.values(), key=lambda spec: spec.name)
    ]


def _schema_description(spec: ToolSpec, workdir: Path | None,
                        coordinator_mode: bool) -> str:
    if spec.name != "agent":
        return spec.description
    try:
        from .agent import agent_tool_description
        return agent_tool_description(workdir, coordinator_mode=coordinator_mode)
    except Exception:
        return spec.description


def _schema_parameters(spec: ToolSpec, coordinator_mode: bool) -> dict:
    if spec.name != "agent" or coordinator_mode:
        return spec.parameters
    params = copy.deepcopy(spec.parameters)
    props = params.get("properties")
    if isinstance(props, dict):
        props.pop("agent_id", None)
    required = params.get("required")
    if isinstance(required, list):
        params["required"] = [name for name in required if name != "agent_id"]
    return params


def find_tool_spec(name: str) -> ToolSpec | None:
    spec = REGISTRY.get(name)
    if spec:
        return spec
    return next((s for s in REGISTRY.values() if name in s.aliases), None)


def available_tool_names() -> str:
    return ", ".join(sorted(REGISTRY))


def validate_tool_input(name: str, arguments: dict, ctx: ToolContext) -> None:
    spec = find_tool_spec(name)
    if spec:
        _validate_runtime_tool_access(spec, arguments, ctx)
    if spec and spec.validate_input:
        spec.validate_input(ctx, arguments)


def tool_property(spec: ToolSpec | None, prop: str, arguments: dict) -> bool:
    if spec is None:
        return False
    raw = getattr(spec, prop)
    if callable(raw):
        try:
            return bool(raw(arguments))
        except Exception:
            return False
    return bool(raw)


def execute_tool(name: str, arguments: dict, ctx: ToolContext) -> ToolResult:
    """统一执行入口：计时、捕获、截断。不存在的工具名也走可恢复错误路径。"""
    t0 = time.monotonic()
    ctx.throw_if_cancelled()
    spec = find_tool_spec(name)
    if spec is None:
        return ToolResult(
            f"Unknown tool '{name}'. Available tools: {available_tool_names()}.",
            ok=False, duration_ms=_ms(t0), error_kind="unknown_tool")
    try:
        validate_tool_input(spec.name, arguments, ctx)
        ctx.progress(phase="started")
        raw = spec.fn(ctx, **arguments)
        ctx.throw_if_cancelled()
        if isinstance(raw, ToolResult):
            result = raw
        else:
            result = ToolResult(str(raw))
        text, truncated, persisted_path = _budget_result(
            result.text, _result_limit(spec, ctx), ctx, spec.name)
        result.text = text
        result.truncated = result.truncated or truncated
        result.persisted_path = result.persisted_path or persisted_path
        if result.truncated:
            ctx.progress(phase="truncated")
        result.duration_ms = _ms(t0)
        return result
    except ToolError as e:
        return ToolResult(str(e), ok=False, duration_ms=_ms(t0), error_kind="tool_error")
    except CancelledError:
        raise
    except TypeError as e:
        # strict 模式下基本不会发生；防御中转网关不支持 strict 的情况
        return ToolResult(f"Invalid arguments for '{name}': {e}", ok=False,
                          duration_ms=_ms(t0), error_kind="invalid_arguments")
    except Exception as e:  # 工具自身 bug 也回传，给模型换路径的机会
        return ToolResult(f"Tool '{name}' crashed: {type(e).__name__}: {e}",
                          ok=False, duration_ms=_ms(t0), error_kind="crash")


def _validate_runtime_tool_access(spec: ToolSpec, arguments: dict,
                                  ctx: ToolContext) -> None:
    allowed = ctx.runtime.allowed_tools
    if allowed is not None and spec.name not in allowed:
        raise ToolError(
            f"Tool '{spec.name}' is not available to this agent. "
            f"Allowed tools: {', '.join(sorted(allowed)) or '(none)'}.")
    if spec.name in ctx.runtime.disallowed_tools:
        raise ToolError(f"Tool '{spec.name}' is disallowed for this agent.")
    if ctx.runtime.require_read_only_tools and not tool_property(spec, "read_only", arguments):
        raise ToolError(
            f"Tool '{spec.name}' is not read-only for these arguments. "
            "Use read_file, grep, glob_files, file_info, show_diff, or a read-only bash command.")


def _result_limit(spec: ToolSpec, ctx: ToolContext) -> int:
    if spec.max_result_size_chars is not None:
        return spec.max_result_size_chars
    return ctx.output_limit


def _budget_result(text: str, limit: int, ctx: ToolContext,
                   tool_name: str) -> tuple[str, bool, str | None]:
    if len(text) <= limit:
        return text, False, None
    path = _persist_result(text, ctx, tool_name)
    if path:
        preview, _ = _truncate(text, limit)
        rel = _display_path(path, ctx.workdir)
        return (f"{preview}\n... [output truncated; full output saved to {rel}. "
                "Use read_file with offset/max_lines or a narrower command to inspect it.] ...",
                True, str(rel))
    truncated, _ = _truncate(text, limit)
    return truncated, True, None


def _truncate(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    head = text[: int(limit * 0.8)]
    tail = text[-int(limit * 0.1):]
    omitted = len(text) - len(head) - len(tail)
    return (f"{head}\n... [output truncated: {omitted} chars omitted; "
            f"narrow the request (offset/max_lines, grep, head) to see more] ...\n{tail}", True)


def _persist_result(text: str, ctx: ToolContext, tool_name: str) -> Path | None:
    try:
        out_dir = ctx.workdir / ".tiny-harness" / "tool-results"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{tool_name}-{uuid.uuid4().hex[:10]}.txt"
        path.write_text(text, encoding="utf-8")
        ctx.runtime.persisted_results[path.name] = str(path)
        return path
    except OSError:
        return None


def _display_path(path: Path, workdir: Path) -> Path:
    try:
        return path.relative_to(workdir)
    except ValueError:
        return path


def _ms(t0: float) -> int:
    return int((time.monotonic() - t0) * 1000)
