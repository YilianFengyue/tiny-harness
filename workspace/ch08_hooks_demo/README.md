# CH08 Hooks Demo

This workspace is a tiny invoice-report project for TUI acceptance.

Suggested prompt:

```text
修复这个项目的金额汇总 bug。要求：先看 README 和测试，修复后运行 python -m pytest，然后写 ACCEPTANCE.md 总结改动和测试结果。
```

What to observe in TUI:

- `/hooks` shows trusted lifecycle hooks.
- `UserPromptSubmit` injects project guidance before the first model call.
- `PreToolUse` changes `python -m pytest` to `python -m pytest -q`.
- `PreToolUse` blocks edits to `src/production_config.py`.
- `PostToolUse` records write/edit audit lines in `.tiny-harness/hook-audit.log`.
- `Stop` asks the model to create `ACCEPTANCE.md` if it tries to finish too early.

Manual hook block check:

```text
请尝试把 src/production_config.py 里的 ENV 改成 prod-test，然后解释发生了什么。
```
