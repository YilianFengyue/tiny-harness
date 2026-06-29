"""工具门控 hook：危险操作的"失败冗长、成功无声"。

命中危险模式时的策略（按序）：
  --yolo          → 放行（但 trajectory 留痕，事后可审计）
  交互式终端       → 当场询问用户
  非交互（eval 等）→ 拒绝，拒绝理由回传模型让它换安全路径
拒绝文案刻意可操作：直接告诉模型"换一种做法"，而不是只说 no。
"""
from __future__ import annotations

import sys

from .config import Config
from .tools.registry import REGISTRY


def gate_tool_call(name: str, arguments: dict, cfg: Config) -> tuple[bool, str | None]:
    """返回 (是否放行, 危险原因——None 表示未命中危险模式)。"""
    spec = REGISTRY.get(name)
    if spec is None or spec.dangerous_check is None:
        return True, None
    why = spec.dangerous_check(arguments)
    if why is None:
        return True, None
    if cfg.yolo:
        return True, why
    if sys.stdin.isatty():
        try:
            answer = input(f"\n[hook] 危险操作（{why}）：{arguments}\n放行？[y/N] ")
            if answer.strip().lower() == "y":
                return True, why
        except EOFError:
            pass
    return False, why


def denial_message(name: str, why: str) -> str:
    return (f"ERROR: tool call blocked by safety policy ({why}). "
            "Do NOT retry the same command rephrased. Choose a safer approach: "
            "operate only on files inside the workspace, avoid destructive/system-level "
            "commands, or accomplish the goal with the dedicated file tools.")
