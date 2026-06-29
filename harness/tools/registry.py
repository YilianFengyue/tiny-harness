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

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


@dataclass
class ToolContext:
    """工具执行环境（loop 注入）。"""
    workdir: Path
    bash_timeout: int = 60
    output_limit: int = 20_000


@dataclass
class ToolResult:
    text: str
    ok: bool = True
    truncated: bool = False
    duration_ms: int = 0


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict
    fn: Callable[..., str]
    dangerous_check: Callable[[dict], str | None] | None = None  # 返回命中原因或 None


REGISTRY: dict[str, ToolSpec] = {}


def tool(name: str, description: str, parameters: dict,
         dangerous_check: Callable[[dict], str | None] | None = None):
    """注册工具。fn 签名为 fn(ctx: ToolContext, **arguments) -> str，
    抛 ToolError 表示可恢复的业务错误（回传模型自纠），其他异常视为 bug。"""
    def deco(fn):
        REGISTRY[name] = ToolSpec(name, description, _strictify(parameters), fn, dangerous_check)
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


def openai_tool_schemas() -> list[dict]:
    return [
        {"type": "function",
         "function": {"name": s.name, "description": s.description,
                      "parameters": s.parameters, "strict": True}}
        for s in REGISTRY.values()
    ]


def execute_tool(name: str, arguments: dict, ctx: ToolContext) -> ToolResult:
    """统一执行入口：计时、捕获、截断。不存在的工具名也走可恢复错误路径。"""
    t0 = time.monotonic()
    spec = REGISTRY.get(name)
    if spec is None:
        return ToolResult(
            f"Unknown tool '{name}'. Available tools: {', '.join(REGISTRY)}.",
            ok=False, duration_ms=_ms(t0))
    try:
        text = spec.fn(ctx, **arguments)
        text, truncated = _truncate(text, ctx.output_limit)
        return ToolResult(text, ok=True, truncated=truncated, duration_ms=_ms(t0))
    except ToolError as e:
        return ToolResult(str(e), ok=False, duration_ms=_ms(t0))
    except TypeError as e:
        # strict 模式下基本不会发生；防御中转网关不支持 strict 的情况
        return ToolResult(f"Invalid arguments for '{name}': {e}", ok=False, duration_ms=_ms(t0))
    except Exception as e:  # 工具自身 bug 也回传，给模型换路径的机会
        return ToolResult(f"Tool '{name}' crashed: {type(e).__name__}: {e}",
                          ok=False, duration_ms=_ms(t0))


def _truncate(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    head = text[: int(limit * 0.8)]
    tail = text[-int(limit * 0.1):]
    omitted = len(text) - len(head) - len(tail)
    return (f"{head}\n... [output truncated: {omitted} chars omitted; "
            f"narrow the request (offset/max_lines, grep, head) to see more] ...\n{tail}", True)


def _ms(t0: float) -> int:
    return int((time.monotonic() - t0) * 1000)
