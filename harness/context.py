"""Context 管理：基于真实 usage 的预算跟踪 + 工具结果清理。

策略选择（DESIGN.md §Context 有完整权衡）：
- 采用【清除旧工具结果】而非摘要压缩：这是改动最小、无信息再加工失真的
  压缩方式（Anthropic context editing API 的默认策略同此，trigger=100K、
  keep=3 对）。摘要压缩作为第二层手段在文档中讨论，本实现刻意不做——
  它引入额外 LLM 调用、摘要失真风险，且会打破前缀缓存。
- 触发依据用 API 返回的真实 prompt_tokens（上一轮），而非本地估算：
  本地 tokenizer 对不上服务端（尤其经中转），真实账单数字才是 ground truth。
- 预算默认 240K：留出余量避开 gpt-5.5 的 272K 长上下文加价线（input x2）。
- 只改写 role=tool 消息的 content，不动消息结构——tool_call_id 链必须完整，
  否则违反 OpenAI 协议。
"""
from __future__ import annotations

from dataclasses import dataclass

PLACEHOLDER = "[tool result cleared to save context: ~{est} tokens. Re-run the tool if needed.]"


@dataclass
class ContextManager:
    budget_tokens: int = 240_000
    keep_recent: int = 3          # 保留最近 N 条工具结果不清理
    last_prompt_tokens: int = 0   # 上一轮真实 input 规模（由 loop 在每次响应后更新）

    def observe(self, prompt_tokens: int) -> None:
        self.last_prompt_tokens = prompt_tokens

    def maybe_compact(self, messages: list[dict]) -> dict | None:
        """超预算时就地清理旧 tool 消息，返回清理统计（无动作返回 None）。"""
        if self.last_prompt_tokens < self.budget_tokens:
            return None

        tool_indices = [i for i, m in enumerate(messages)
                        if m.get("role") == "tool" and not m.get("_cleared")]
        clearable = tool_indices[:-self.keep_recent] if self.keep_recent else tool_indices
        if not clearable:
            return None

        cleared, freed_chars = 0, 0
        for i in clearable:
            content = messages[i].get("content") or ""
            if len(content) < 200:    # 太短的清了也省不出什么，还破坏可读性
                continue
            est = max(len(content) // 4, 1)   # chars/4 粗估，只用于占位符提示
            messages[i]["content"] = PLACEHOLDER.format(est=est)
            messages[i]["_cleared"] = True    # 发送前由 loop 剥除的内部标记
            cleared += 1
            freed_chars += len(content)

        if cleared == 0:
            return None
        # 乐观下调水位，避免下一轮重复触发；真实值由下一次 API 响应矫正
        est_freed = freed_chars // 4
        self.last_prompt_tokens = max(self.last_prompt_tokens - est_freed, 0)
        return {"cleared_messages": cleared, "est_tokens_freed": est_freed}


def strip_internal_marks(messages: list[dict]) -> list[dict]:
    """发送给 API 前剥除内部标记字段（协议不认识多余字段）。"""
    return [{k: v for k, v in m.items() if not k.startswith("_")} for m in messages]
