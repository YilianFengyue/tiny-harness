# tiny-harness 完整架构文档

> 本文档面向第一次接触本项目的读者，目标是：读完后无需翻源码即可理解项目做了什么、
> 怎么做的、为什么这样做。适合发给 ChatGPT / Claude 等大模型作为上下文窗口参考。

---

## 0. 项目定位

**题目要求**：不用任何 agent / harness 框架（LangChain、inspect_ai 等），用纯 Python 实现：
1. Agent loop（停止条件、最大轮数、工具报错恢复）
2. ≥3 个工具（calculator、文件读写、bash）
3. Context 管理（token 跟踪、超阈值策略）
4. Infra & 可观测性（trajectory 日志、API 重试、token/成本统计、唯一 run_id）

最终用"读 CSV 算均值写文件"等任务验证。

**最终成果**：~2300 行 Python + 1575 行零依赖 HTML Viewer，pytest 40/40，真实 API 48/48 零失败。

**核心立场**：loop 极简（~80 行核心），复杂度全部投入 harness 基础设施——这个配比本身就是对"harness 工程"命题的回答。

---

## 1. 文件结构

```
tiny-harness/
├── main.py                          # CLI 入口（129 行）
├── demo.py                          # 生成示例 data.csv（20 行）
├── requirements.txt                 # 仅 openai>=1.60 + pytest>=8.0
├── .env                             # API key / 模型 / base_url 配置
│
├── harness/                         # ===== 核心框架 =====
│   ├── __init__.py
│   ├── config.py                    # 配置加载、价格表、常量（87 行）
│   ├── loop.py                      # Agent 主循环（190 行，核心 ~80 行）
│   ├── context.py                   # Context 预算管理与清理（63 行）
│   ├── telemetry.py                 # JSONL 事件日志 + 成本台账（158 行）
│   ├── hooks.py                     # 危险操作安全门控（42 行）
│   ├── skills.py                    # 领域知识注入（37 行）
│   │
│   ├── providers/                   # 模型协议抽象层
│   │   ├── base.py                  # 抽象基类 Provider + 数据类（83 行）
│   │   └── openai_chat.py           # OpenAI Chat Completions 实现 + 离线重放（156 行）
│   │
│   └── tools/                       # 工具定义与注册表
│       ├── __init__.py              # 导入即注册
│       ├── registry.py              # @tool 装饰器、strict schema、统一执行与截断（118 行）
│       ├── calculator.py            # AST 白名单安全求值（95 行）
│       ├── files.py                 # read_file / write_file / list_files（129 行）
│       └── bash.py                  # Shell 执行 + 超时 + 危险模式检测（119 行）
│
├── tests/                           # ===== 测试套件（40 个用例，全离线） =====
│   ├── conftest.py                  # MockProvider + 公共 fixture（63 行）
│   ├── test_protocol.py             # 协议正确性：并行调用、坏 JSON、length 处理（150+ 行）
│   ├── test_calculator.py           # AST 安全、注入防御、溢出（47 行）
│   ├── test_files_sandbox.py        # 路径逃逸、符号链接、分页（43 行）
│   ├── test_bash_sandbox.py         # 超时、危险模式、进程树击杀
│   ├── test_retry.py                # 429/5xx/4xx 重试矩阵、Retry-After（92 行）
│   └── test_context_and_cost.py     # Context 清理、长上下文加价、成本公式（81 行）
│
├── eval/                            # ===== 评测系统 =====
│   ├── run_eval.py                  # Dataset→Solver→Scorer 框架（194 行）
│   └── tasks/                       # 6 个 benchmark 任务
│       ├── 01_csv_mean/             # 基本端到端
│       ├── 02_wrong_filename/       # 错误恢复（文件名不匹配）
│       ├── 03_dirty_data/           # 脏数据清洗
│       ├── 04_tool_chain/           # 多工具协作链
│       ├── 05_big_file/             # 30K 行大文件处理
│       └── 06_context_pressure/     # 逼出 context 清理
│
├── skills/
│   └── csv-data-processing.md       # CSV 处理领域经验（7 条规则）
│
├── viewer/
│   └── index.html                   # 零依赖 Trajectory 可视化器（1575 行）
│
├── runs/                            # 运行输出（运行时生成）
│   └── <run_id>/
│       ├── trajectory.jsonl         # 完整事件流
│       └── summary.json             # 终局统计
│
├── DESIGN.md                        # 设计决策与权衡论述
├── RETROSPECTIVE.md                 # 全程复盘报告
└── ARCHITECTURE.md                  # 本文档
```

---

## 2. 整体架构图

```
                            ┌─────────────┐
                            │   main.py   │  CLI 入口 / serve / replay / resume
                            │   (入口)    │
                            └──────┬──────┘
                                   │ 构造 Config, Provider, RunLogger
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│                          loop.py  (Agent 主循环)                     │
│                                                                      │
│  for turn in 1..max_turns:                                           │
│    ┌─ 成本熔断检查 (ledger.cost_usd >= max_cost_usd)                 │
│    ├─ Context 清理 (cm.maybe_compact)                                │
│    ├─ 调模型 (provider.complete) ──────────┐                         │
│    ├─ 记账 (ledger.record)                 │                         │
│    ├─ 日志 (logger.emit)                   │                         │
│    └─ 分派:                                │                         │
│       ├─ 有 tool_calls → 执行工具 → 继续    │                         │
│       ├─ length → nudge/truncated          │                         │
│       └─ stop → completed                  │                         │
└───────┬─────────────┬──────────────────────┘                         │
        │             │                                                │
        ▼             ▼                                                ▼
┌──────────────┐ ┌──────────────┐                        ┌────────────────────┐
│  tools/      │ │  context.py  │                        │  providers/        │
│  registry    │ │  预算管理    │                        │  ├ base.py (ABC)   │
│  ├calculator │ │  清理旧结果  │                        │  ├ openai_chat.py  │
│  ├files      │ └──────────────┘                        │  │  ├ 自研重试     │
│  └bash       │                                         │  │  └ 方言处理     │
│    ↓         │                                         │  └ ReplayProvider  │
│ hooks.py     │                                         │    (离线重放)      │
│ (安全门控)   │                                         └────────────────────┘
└──────────────┘
        │                        ┌────────────────────┐
        │                        │  telemetry.py      │
        └───── 所有事件 ────────→│  ├ RunLogger       │──→ runs/<id>/trajectory.jsonl
                                 │  ├ CostLedger      │──→ runs/<id>/summary.json
                                 │  └ Usage           │
                                 └────────────────────┘
```

---

## 3. 核心层详解

### 3.1 Agent Loop（harness/loop.py）

**职责**：调模型 → 按 finish_reason 分派 → 执行工具 → 回传 → 重复。

**关键函数**：

| 函数 | 行数 | 作用 |
|------|------|------|
| `run_agent()` | 54-124 | 主循环，唯一的公开 API |
| `_run_tool_calls()` | 127-155 | 执行一轮全部工具调用，为每个 tool_call_id 生成应答 |
| `tool_message()` | 158-160 | 构造 role=tool 消息，错误时加 "ERROR: " 前缀 |
| `build_resume_messages()` | 163-181 | 从 trajectory 重建消息历史，实现断点续跑 |
| `build_initial_messages()` | 47-51 | 构造初始 system + user 消息 |

**三个协议铁律**（都有专门的测试覆盖）：

1. **每个 tool_call_id 必须应答**：模型发出的每一个 tool_call，不管是正常执行、执行出错、参数 JSON 非法、还是被安全策略拒绝，都必须回一条 `role=tool` 消息。漏答任何一个，下次 API 请求直接 400。

2. **工具错误回传而非终止**：Agent 的本质是靠环境反馈自纠。工具报错时不终止循环，而是把错误信息（"ERROR: " 前缀）发回模型，让它自己换策略。OpenAI 协议没有 is_error 字段，前缀是给模型的无歧义信号。

3. **finish_reason == "length" 不是完成**：这意味着模型的输出被 token 上限截断了。第一次截断时发一条 nudge 消息提示模型收敛（把大内容写到文件而非回复里），第二次仍截断才认定为 "truncated" 终止。

**三重熔断**：

```python
# 1. 轮数熔断
for turn in range(1, cfg.max_turns + 1):

# 2. 成本熔断（每轮循环开头检查）
if ledger.cost_usd >= cfg.max_cost_usd:
    reason = "max_cost"
    break

# 3. 中断熔断（Ctrl-C 优雅处理，仍写 run_end 事件）
except KeyboardInterrupt:
    reason = "interrupted"
```

**终止原因**枚举：`completed`（正常完成）| `max_turns` | `max_cost` | `truncated` | `interrupted` | `error`

**并行工具执行**：当模型一次返回多个 tool_call 时，用 ThreadPoolExecutor 真并行执行：

```python
if len(tool_calls) > 1:
    with ThreadPoolExecutor(max_workers=min(len(tool_calls), 8)) as pool:
        results = list(pool.map(run_one, tool_calls))
```

**System Prompt 设计**：

```
You are a precise, autonomous agent working in a sandboxed workspace.
Workspace root: {workdir}
Operating rules:
- Use tools for every action and computation; never invent file contents or numeric results.
- Verify deliverables: after writing an output file, read it back to confirm.
- Double-check arithmetic with the calculator tool.
- If a tool returns an error, read it carefully and adapt instead of repeating.
- When done, reply with a concise final summary (no tool calls).
{skills}  ← 可选的领域知识注入
```

### 3.2 Provider 抽象层（harness/providers/）

**设计原则**：loop 完全不碰任何厂商 SDK 的类型。通过两个数据类解耦：

#### ToolCallRequest（模型发起的工具调用）

```python
@dataclass
class ToolCallRequest:
    id: str                         # 协议要求的唯一 ID
    name: str                       # 工具名
    arguments_raw: str              # 原始 JSON 字符串（重放保真）
    arguments: dict | None = None   # 解析结果；None = 非法 JSON
    parse_error: str | None = None  # JSON 解析错误信息
```

`arguments_raw` 保留原始字符串有两个原因：
- **重放保真**：ReplayProvider 需要原始数据
- **协议需要原文**：assistant 消息里的 tool_calls 必须用原始 arguments 字符串

`from_raw()` 工厂方法统一处理解析，JSON 无效时设 `parse_error` 而非抛异常——因为这是一种需要回传给模型的"工具错误"。

#### ModelTurn（模型响应的厂商无关表示）

```python
@dataclass
class ModelTurn:
    content: str | None                            # 文本回复
    tool_calls: list[ToolCallRequest]              # 工具调用请求
    finish_reason: str = "stop"                    # 归一化: "tool_calls" | "stop" | "length"
    usage: Usage                                   # token 用量
    request_id: str | None = None                  # API 请求 ID（debug 用）
    latency_ms: int = 0                            # 响应延迟
    reasoning_content: str | None = None           # DeepSeek 思考模型的方言字段
```

`reasoning_content` 是 DeepSeek 等思考模型的协议方言：响应里的推理内容必须原样回传到后续请求的 assistant 消息中，否则 400 报错。标准 OpenAI 模型没有该字段，None 时不写入消息。

#### OpenAIChatProvider

**重试策略**（自研，关掉 SDK 内置的 `max_retries=0`）：

```
可重试：429（限流）+ 全部 5xx（含 Cloudflare 52x 系列）
不可重试：400 / 401 / 403 / 404 / 413 / 422
退避公式：sleep = min(2^attempt + U(0,1), 60s)
```

- 抖动（`+ U(0,1)`）防止 thundering herd：同时被 429 打回的多个客户端不会同步重试
- 每次退避写进 trajectory（`on_retry` 回调）——这是自研而非用 SDK 的唯一原因：**可观测性优先**

**DeepSeek 方言处理**：

```python
self._thinking_dialect = False  # 初始关闭

# 当某次响应携带 reasoning_content 时，开启方言模式
if reasoning_content is not None:
    self._thinking_dialect = True

# 开启后，补全历史 assistant 消息缺失的 reasoning_content 字段
if self._thinking_dialect:
    messages = [
        {**m, "reasoning_content": m.get("reasoning_content", "")}
        if m.get("role") == "assistant" else m
        for m in messages
    ]
```

这里有一个在真实 API 调试中发现的细节：不是每一轮模型都会输出 reasoning_content，但一旦某次输出了，**后续所有** assistant 消息都必须带该字段（缺失的补空串），否则第 N 轮才会 400。这就是 RETROSPECTIVE 里 bug #4 的故事。

#### ReplayProvider

从历史 trajectory.jsonl 中提取 `llm_response` 事件，按顺序重放：

```python
class ReplayProvider(Provider):
    def __init__(self, events: list[dict]):
        self._responses = [e for e in events if e["type"] == "llm_response"]
        self._i = 0

    def complete(self, messages, tools, on_retry=None) -> ModelTurn:
        e = self._responses[self._i]  # 取下一个录制的响应
        self._i += 1
        return ModelTurn(...)         # 从事件字段重建
```

一个 Provider 同时解决三件事：
- **离线协议测试**：不花钱跑完整 agent loop
- **面谈现场演示**：零成本展示完整运行过程
- **历史运行复现**：验证 trajectory 记录的完整性

### 3.3 工具系统（harness/tools/）

#### Registry（registry.py）— 工具注册与执行框架

**@tool 装饰器**：注册工具到全局 `REGISTRY` 字典：

```python
REGISTRY: dict[str, ToolSpec] = {}  # {工具名: ToolSpec}

@dataclass
class ToolSpec:
    name: str
    description: str      # 给模型看的工具说明
    parameters: dict      # JSON Schema
    fn: Callable          # fn(ctx: ToolContext, **arguments) -> str
    dangerous_check: ...  # 返回危险原因或 None
```

**strict schema 自动补全**（`_strictify()`）：

OpenAI 的 strict 模式要求每层 object 都有 `additionalProperties: false` 且字段全部在 `required` 列表里。`_strictify()` 递归补全这些字段，让工具定义时不用手写：

```python
def _strictify(params: dict) -> dict:
    if params.get("type") == "object":
        params["required"] = list(params["properties"].keys())
        params["additionalProperties"] = False
        for sub in params["properties"].values():
            _strictify(sub)
    # ...
```

效果：API 层面消灭参数幻觉——模型不会传出 schema 之外的字段。

**统一执行入口**（`execute_tool()`）：

```
1. 查找 REGISTRY 中的工具（不存在 → 返回可恢复错误）
2. 调用 spec.fn(ctx, **arguments)
3. 截断过长输出 → ToolResult
4. 捕获所有异常 → 回传模型而非崩溃
```

异常分三级处理：
- `ToolError`：可恢复的业务错误（文件不存在、表达式语法错误等）
- `TypeError`：参数类型不匹配（strict 模式下基本不会发生）
- `Exception`：工具自身 bug，也回传给模型让它换路径

**输出截断策略**：默认 20K 字符上限，超出时保留 80% head + 10% tail + 导航提示：

```
{前 80% 的内容}
... [output truncated: 15000 chars omitted;
narrow the request (offset/max_lines, grep, head) to see more] ...
{后 10% 的内容}
```

截断信息本身会告诉模型"怎么缩小请求"——不只是说"被截了"。

#### calculator（calculator.py）— AST 白名单安全求值

**为什么不用 `eval()`**：eval 的沙箱化是已被反复证伪的方向（`__builtins__` 注入、dunder 链），AST 白名单从根本上只允许算术结构存在。

实现方式：`ast.parse(expression, mode="eval")` 解析表达式，然后递归遍历 AST 节点，只允许以下结构：

| AST 节点类型 | 允许的内容 |
|---|---|
| `ast.Constant` | int / float（排除 bool、字符串等） |
| `ast.BinOp` | `+ - * / // % **` |
| `ast.UnaryOp` | `+x` / `-x` |
| `ast.Name` | 常量 `pi` / `e` |
| `ast.Call` | 白名单函数：abs, round, min, max, sum, sqrt, log, log10, log2, exp, floor, ceil, sin, cos, tan |
| `ast.List/Tuple` | 仅供 `sum([1,2,3])` / `min(1,2)` 使用 |

**指数溢出保护**：

```python
def _safe_pow(a, b):
    if abs(b) > 10_000 or (abs(a) > 1 and abs(b) * math.log10(abs(a)) > 308):
        raise ToolError("exponent too large; result would overflow")
    return a ** b
```

任何不在白名单中的语法（变量赋值、字符串、import 等）都会被拒绝并返回明确的错误信息。

#### files（files.py）— 文件操作三件套

**路径安全**：核心是 `resolve_in_workdir()` 函数：

```python
def resolve_in_workdir(ctx, path):
    candidate = (Path(path) if Path(path).is_absolute() else ctx.workdir / path).resolve()
    workdir = ctx.workdir.resolve()
    if candidate != workdir and workdir not in candidate.parents:
        raise ToolError("path escapes the workspace...")
```

为什么在 `resolve()` 之后判定而非字符串层面检查 `".."`：
- 符号链接可以指向 workdir 外
- Windows 绝对路径（`C:\`）绕过相对路径检查
- 大小写、UNC 路径都能绕过字符串过滤
- **`resolve()` 会消解所有这些**，在最终绝对路径上判定才可靠

三个工具：

| 工具 | 功能 | 关键细节 |
|------|------|---------|
| `read_file` | 带行号的分页读取 | offset（起始行）+ max_lines（默认 500），尾部提示"还有 N 行，continue with offset=X" |
| `write_file` | 写文件 | 自动创建父目录 |
| `list_files` | 列目录 | 非递归，显示类型和大小，上限 200 条 |

**错误信息设计**——文件不存在时：

```
file 'data.csv' not found. Files in '.': sales_data.csv, readme.txt
```

直接列出目录内容，让模型看到正确的文件名——这不是简单的 "File not found"。

#### bash（bash.py）— Shell 执行沙箱

**四层防护**：

1. **cwd 锁定**：`subprocess.Popen(cwd=str(ctx.workdir))` —— 命令在 workdir 内执行
2. **超时击杀**：`proc.communicate(timeout=ctx.bash_timeout)` + 超时时杀整棵进程树
3. **输出截断**：由 registry 的统一截断保证
4. **危险模式检测**：正则匹配 → 交给 hooks.py 决策

**危险模式表**（8 个正则）：

```python
DANGEROUS_PATTERNS = [
    (r"\brm\s+(-[a-z]*[rf]...", "recursive delete near filesystem root"),
    (r"\bmkfs\b|\bdd\s+...",    "raw disk write"),
    (r"\b(shutdown|reboot)\b",  "system power control"),
    (r":\(\)\s*\{.*\}\s*;\s*:", "fork bomb"),
    (r"\bsudo\b",               "privilege escalation"),
    (r"(curl|wget).*\|\s*sh",   "pipe remote script into shell"),
    (r">\s*/dev/sd[a-z]",       "raw device overwrite"),
    (r"\bgit\s+push.*--force",  "force push"),
]
```

**进程树击杀**（跨平台）：

```python
def _kill_tree(proc):
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)])
    else:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
```

Windows 下 `proc.kill()` 只杀直接子进程，`taskkill /T` 杀整棵树。POSIX 下用 `start_new_session=True` 创建进程组再 `killpg`。

**诚实的安全声明**：文档和代码注释里明确声明——字符串过滤不是安全边界（base64 编码、变量拼接、子 shell 都能绕过）。本项目的威胁模型是"模型犯傻"而非"模型作恶"，软沙箱 + 明示边界是匹配该威胁模型的工程选择。

### 3.4 安全门控（harness/hooks.py）

当工具的 `dangerous_check` 命中危险模式时，按优先级决策：

```
--yolo 标志     → 放行（trajectory 留痕，事后可审计）
交互式终端       → 当场询问用户 [y/N]
非交互（eval 等）→ 拒绝
```

拒绝文案设计——**可操作**而非简单说 no：

```
ERROR: tool call blocked by safety policy (recursive delete near filesystem root).
Do NOT retry the same command rephrased. Choose a safer approach:
operate only on files inside the workspace, avoid destructive/system-level
commands, or accomplish the goal with the dedicated file tools.
```

### 3.5 Context 管理（harness/context.py）

**策略**：清除旧的 tool result 内容，**不做 LLM 摘要**。

为什么不做摘要：
- 清除无信息失真风险
- 不需要额外 API 调用（省钱）
- 不破坏前缀缓存（缓存靠稳定的消息前缀命中）
- 这也是 Anthropic context editing API 的默认策略

**触发条件**：API 返回的真实 `prompt_tokens` 超过预算（默认 240K）。

为什么用 API 返回值而非本地 tiktoken 估算：经中转网关后本地 tokenizer 对不上服务端（可能是不同模型），API 账单数字才是 ground truth。

**预算为什么是 240K**：gpt-5.5 单次请求 input 超过 272K 时触发长上下文加价（input ×2 / output ×1.5）。240K 在加价线前留出 32K 余量。这不是拍脑袋的数字，是经济学硬约束。

**清理算法**：

```python
def maybe_compact(self, messages):
    if self.last_prompt_tokens < self.budget_tokens:
        return None  # 未超预算，不动

    # 找出所有 role=tool 且未清理过的消息
    tool_indices = [i for i, m in enumerate(messages) if m["role"] == "tool" and not m["_cleared"]]

    # 保留最近 N 条不清理
    clearable = tool_indices[:-self.keep_recent]

    for i in clearable:
        content = messages[i]["content"]
        if len(content) < 200:     # 太短的不清理（收益太小）
            continue
        est = max(len(content) // 4, 1)
        messages[i]["content"] = f"[tool result cleared to save context: ~{est} tokens. Re-run the tool if needed.]"
        messages[i]["_cleared"] = True  # 内部标记，发送 API 前会被 strip_internal_marks() 剥除
```

关键约束：**只改 content 字段，不删消息**——`tool_call_id` 链必须完整，否则违反 OpenAI 协议。

### 3.6 可观测性（harness/telemetry.py）

#### run_id 与目录结构

```python
def new_run_id():
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
# 示例：20260612-143021-a1b2c3
```

每次运行生成独立目录：`runs/20260612-143021-a1b2c3/`，包含 `trajectory.jsonl` + `summary.json`。

#### 9 种事件类型

```
run_start      {task, model, workdir, config, sdk_version, skills}
llm_request    {turn, model, n_messages, messages(完整!), tools, params}
llm_response   {turn, finish_reason, content, tool_calls, usage, cost_usd, request_id, latency_ms, reasoning_content}
retry          {turn, attempt, status, error, sleep_s}
tool_call      {turn, tool_call_id, name, arguments}
tool_result    {turn, tool_call_id, name, ok, result, duration_ms, truncated}
context_edit   {turn, cleared_messages, est_tokens_freed, prompt_tokens_before}
error          {where, error, traceback}
run_end        {reason, turns, usage_total, cost_usd, pricing_unknown, duration_s, final_message}
```

**每个事件都有**：`run_id` + 单调递增 `step` + ISO 8601 UTC 时间戳。

**llm_request 记录完整 messages** 是核心设计决策：LLM 推理本质不确定（服务端动态 batching，temperature=0 也不保证），完整请求/响应是唯一可靠的重放凭据。这是"记录而非控制"的可复现策略。

**JSONL 格式的优势**：
- append-only：崩溃时最多丢正在写的一行（每条后 flush）
- 可流式 tail
- 每行自包含，一行损坏不影响其他行

#### 成本台账（CostLedger）

```python
cost = (prompt_tokens - cached_tokens) × P_input     # 新 input
     + cached_tokens × P_cached                        # 缓存命中（通常 input 的 10%）
     + completion_tokens × P_output                    # output（含 reasoning）
```

**长上下文加价**：单次请求 prompt_tokens > 272K 时，`input × 2.0` / `output × 1.5`。

**未知模型**：不编造价格——设 `pricing_unknown = True` 并在 summary 里标记，成本按 0 计。

价格表支持通过环境变量 `TINY_HARNESS_PRICING` 覆盖（JSON 格式），适配自建模型或新模型。

### 3.7 领域知识注入（harness/skills.py）

Skills 是纯 markdown 文件，注入到 system prompt 尾部：

```
# Domain knowledge
Expert guidance for this kind of task. Follow it unless it conflicts
with the user's explicit instructions.

## CSV / 表格数据处理经验
1. 动手前先看数据：用 bash 跑 head -5 确认表头...
2. 列序号按自然语言习惯是 1-based...
3. 大于几百行就别用 read_file 全量读...
...
```

**棘轮原则**：每条 skill 规则都应追溯到一类真实观察到的 agent 失败模式。不是泛泛的"best practice"，而是从失败中提炼的具体经验。

这是"便宜模型 + 领域经验 ≈ 贵模型"假设的最小实现。

### 3.8 配置层（harness/config.py）

**极简 .env 解析**（15 行，不引入 python-dotenv）：

```python
def load_dotenv(path=None):
    for line in path.read_text().splitlines():
        key, _, value = line.partition("=")
        if key and key not in os.environ:  # 不覆盖已有环境变量
            os.environ[key] = value
```

**Config 数据类**：

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `model` | "gpt-5.5" | 模型名 |
| `max_turns` | 30 | 最大循环轮数 |
| `max_cost_usd` | 1.0 | 美元成本熔断线 |
| `context_budget` | 240,000 | input token 预算 |
| `context_keep_recent` | 3 | 清理时保留最近 N 条工具结果 |
| `tool_output_limit` | 20,000 | 工具输出截断阈值（字符） |
| `bash_timeout` | 60 | bash 命令超时（秒） |
| `max_retries` | 5 | API 重试次数 |
| `yolo` | False | 跳过危险命令确认 |
| `skills` | [] | 注入的 skill 名称列表 |
| `reasoning_effort` | None | none/low/medium/high/xhigh |
| `max_completion_tokens` | None | 推理模型需给推理留 ≥25K |

---

## 4. CLI 使用方式（main.py）

```bash
# 基本用法：给任务描述，agent 自动执行
python main.py "读 data.csv，算第三列的均值，写到 mean.txt" --workdir ./workspace

# 离线重放（不打 API、零成本，从 trajectory 重放模型响应）
python main.py --replay 20260612-143021-a1b2c3

# 断点续跑（从某次运行的消息现场继续，加新的指令）
python main.py --resume 20260612-143021-a1b2c3 "继续：把结果四舍五入到两位小数"

# 启动 Viewer 服务
python main.py serve [port]
# 然后访问 http://localhost:8765/viewer/index.html?file=/runs/<run_id>/trajectory.jsonl

# 带 skill 运行
python main.py "读 data.csv 算均值" --skill csv-data-processing

# 调整参数
python main.py "..." --model gpt-5.5-mini --max-turns 15 --max-cost 0.5 --yolo
```

运行结束输出：

```
结束原因: completed    轮数: 6    耗时: 12.3s
tokens: input=4521 (cached 1200)  output=892 (reasoning 54)
成本: $0.0432
trajectory: runs/20260612-143021-a1b2c3/trajectory.jsonl
可视化: python main.py serve → http://localhost:8765/viewer/...
```

---

## 5. 评测系统（eval/）

### 5.1 架构：Dataset → Solver → Scorer

```
eval/run_eval.py
    │
    ├─ discover_tasks()         扫描 eval/tasks/*/，找到所有 task.json
    │
    ├─ run_once(task_dir, model, skill, ...)
    │   ├─ 加载 task.json        {"prompt": "...", "max_turns": 15}
    │   ├─ 执行 gen.py           generate(workdir) → 确定性生成输入文件（固定 seed）
    │   ├─ 子进程启动 main.py    agent 执行任务（环境隔离 + 900s 超时熔断）
    │   ├─ 读 summary.json       从 harness 自己的可观测性输出取结果
    │   └─ 执行 scorer.py        score(workdir) → (ok: bool, note: str) 程序化判分
    │
    └─ aggregate(rows)           生成 markdown 汇总报告
```

**子进程隔离**的好处：
- 一个 agent 崩溃不影响整个 eval
- 900s 超时是熔断兜底
- eval 消费的就是 harness 自己的 `summary.json`——闭环自洽

### 5.2 六个 Benchmark 任务

| # | 目录名 | 任务 Prompt | 考察能力 | 关键挑战 |
|---|--------|-------------|---------|---------|
| 01 | `01_csv_mean` | 读 data.csv，算第三列均值，写到 mean.txt | 基本端到端 | 60 行干净 CSV，baseline |
| 02 | `02_wrong_filename` | 同上，但实际文件叫 sales_data.csv | 错误恢复 | read_file 报错后能否 list_files 找到正确文件名 |
| 03 | `03_dirty_data` | 同上，第三列混入 N/A、空值、非数字 | 数据清洗 | 剔除无效值（不是当 0），分子分母都要正确 |
| 04 | `04_tool_chain` | bash 生成 1-100 → 算平方和 → calculator 验证 → 写 result.txt | 多工具协作 | 三个工具串联：bash → calculator → write_file |
| 05 | `05_big_file` | 30K 行 CSV 算均值 | 大文件处理 | read_file 分页读不完，得用 bash + awk/python 流式处理 |
| 06 | `06_context_pressure` | 3 个 CSV 分别算均值，context 预算压到 6K | Context 清理 | 强制触发 maybe_compact，验证清理后任务仍能完成 |

每个任务目录包含：
- `task.json`：prompt + max_turns + 可选的 extra_args
- `gen.py`：`generate(workdir)` 确定性生成输入数据（固定 random seed）
- `scorer.py`：`score(workdir) -> (bool, str)` 程序化判分

**Scorer 设计**——以 01 为例：

```python
def score(workdir):
    out = workdir / "mean.txt"
    if not out.exists():
        return False, "mean.txt 不存在"
    got = extract_number(out.read_text())   # 从文件中提取数字
    want = expected_mean(workdir / "data.csv")  # 独立计算正确答案
    ok = abs(got - want) <= max(abs(want) * 1e-4, 0.01)  # 容忍合理舍入
    return ok, f"got={got} want={want:.4f}"
```

### 5.3 2×2 实验矩阵

```bash
# 完整 2×2 矩阵：{主模型, 便宜模型} × {裸, +skill}，每格重复 2 次
python eval/run_eval.py --matrix --runs 2

# 单模型单 skill
python eval/run_eval.py --model gpt-5.5 --skill csv-data-processing --runs 3

# 只跑部分任务
python eval/run_eval.py --tasks 01,02
```

实验输出到 `eval/results/<timestamp>/`：
- `results.json`：每次运行的详细数据
- `report.md`：markdown 汇总表

---

## 6. 数据流：一次完整运行的生命周期

以"读 data.csv，算第三列的均值，写到 mean.txt"为例：

```
[Turn 1] 用户任务 → 模型收到 system + user prompt
         ← 模型返回 tool_calls: [read_file("data.csv")]
         → 执行 read_file → 返回带行号的 CSV 内容
         ← tool result 追加到 messages

[Turn 2] 模型收到完整历史
         ← 返回 tool_calls: [calculator("(12.5 + 7.3 + ...) / 60")]
         → 执行 calculator → "... = 42.35"
         ← tool result 追加

[Turn 3] 模型收到完整历史
         ← 返回 tool_calls: [write_file("mean.txt", "42.35")]
         → 执行 write_file → "wrote 5 chars to mean.txt"
         ← tool result 追加

[Turn 4] 模型收到完整历史
         ← 返回 tool_calls: [read_file("mean.txt")]  ← 验证写入结果
         → 执行 read_file → "42.35"

[Turn 5] 模型确认结果正确
         ← finish_reason="stop", content="任务完成。均值为 42.35，已写入 mean.txt。"
         → reason="completed", 写 run_end 事件, 生成 summary.json
```

每一步都会 emit 对应的 telemetry 事件到 `trajectory.jsonl`。

---

## 7. 关键设计决策表

| # | 决策 | 备选方案 | 为什么这样选 |
|---|------|---------|-------------|
| 1 | loop 只认 finish_reason，全分支处理 | 只判断"有没有 tool_calls" | `length` 不是完成，`max_tokens` 截断要 nudge 再判 |
| 2 | 工具错误回传而非终止 | 异常直接 raise | agent 靠环境反馈自纠；"ERROR:" 前缀给无歧义信号 |
| 3 | 每个 tool_call_id 必须应答 | 失败的跳过 | 漏答 → 下次请求 400，协议铁律 |
| 4 | 三重熔断 | 只有 max_turns | 成本是独立的失控维度；中断也要写 run_end |
| 5 | strict: true + 自动补全 schema | 裸 JSON Schema | API 层面消灭参数幻觉 |
| 6 | 截断带导航提示 | 硬截断 | 告诉模型怎么缩小请求 |
| 7 | calculator 用 AST 白名单 | eval() | eval 沙箱化被反复证伪 |
| 8 | 路径校验在 resolve() 后 | 字符串查 ".." | symlink/绝对路径绕过字符串过滤 |
| 9 | 危险命令 hook + 明示安全边界 | 上 Docker | 威胁模型是"模型犯傻"不是"模型作恶" |
| 10 | context 清除旧工具结果 | LLM 摘要压缩 | 无失真、不付额外调用、不破缓存 |
| 11 | 触发用 API 真实 prompt_tokens | 本地 tiktoken | 经中转 tokenizer 对不上，账单是 ground truth |
| 12 | JSONL 记完整 messages | 只记摘要 | 可复现 = 记录而非控制（temperature=0 不保证确定性） |
| 13 | --replay 做成一等公民 | 只留日志 | 一个开关解决离线测试+演示+复现 |
| 14 | provider 抽象层 | loop 里直接调 SDK | 换协议=新增一个文件；DeepSeek 方言验证了抽象的价值 |
| 15 | 自研重试 | 用 SDK max_retries | 每次退避要写进 trajectory——可观测性优先 |
| 16 | eval 子进程跑 agent | 进程内调用 | 环境隔离+熔断；eval 消费 harness 的 summary.json，闭环自洽 |
| 17 | 不做 MCP / 多 agent / 摘要压缩 | 全堆上 | 与"无框架理解机制"的考核目标相反，且需要回答"它检验什么假设" |

---

## 8. 真实 API 踩坑记录

这些 bug 全部发生在"40 个离线测试全绿"之后——离线测试保证协议是对的，但真实世界的网关方言只有真跑才能暴露。

| # | 现象 | 根因 | 修复 | 教训 |
|---|------|------|------|------|
| 1 | 连不上 API | .env 的 base_url 缺 `/v1` | 补 `/v1` | 中转配置的第一检查项 |
| 2 | 403「令牌额度不足预扣费」| 中转按 max_output 预扣，key 设了限额 | 用户提额 | 期间实测 `--resume` 断点续跑成功，模型醒来第一件事用 calculator 验算之前的结果 |
| 3 | DeepSeek 400 | `reasoning_content must be passed back` | ModelTurn 增加字段，全链路保真 | 厂商方言是 provider 层的职责 |
| 4 | 修完 #3，第 7 轮还是 400 | 没思考的轮次不返回 reasoning_content，但回传时必须补空串 | 方言开关 + 补 "" | **修 bug 要修到"为什么第 7 轮才炸"水落石出** |
| 5 | Cloudflare 523 直接放弃 | 可重试集合枚举了 {429,500,502,503,504}，没料到 52x | 改为 429 + 全部 5xx | 经中转的现实世界比官方文档更野 |
| 6 | eval 启动即崩 UnicodeEncodeError | GBK 控制台 vs ▶ 字符 | 统一 UTF-8 reconfigure | 崩在第一行 print，一分钱没烧 |

---

## 9. 实验结果

### 9.1 2×2 矩阵（6 任务 × 2 模型 × 2 skill × 2 重复 = 48 次运行）

| 模型 | Skill | 成功率 | 平均轮数 | 平均成本 $ | 平均 reasoning tok | 缓存命中 tok |
|------|-------|--------|---------|-----------|-------------------|-------------|
| gpt-5.4-mini | 无 | 12/12 | 5.7 | 0.0066 | 132 | 4139 |
| gpt-5.4-mini | +skill | 12/12 | 6.1 | 0.0065 | 152 | 6144 |
| gpt-5.5 | 无 | 12/12 | 6.3 | 0.0432 | 54 | 1664 |
| gpt-5.5 | +skill | 12/12 | 7.2 | 0.0561 | 34 | 5803 |

### 9.2 核心结论

- **48/48 天花板效应**：任务太简单，差分不出 skill 增益。诚实承认："便宜+经验≈旗舰"的假设**未被检验**（不是被证实）
- **但天花板给出更强结论**：harness 质量主导该任务类——mini 以 **1/6.5 成本**追平旗舰，包含脏数据清洗和文件名自恢复全部正确

### 9.3 三个意外观测

1. **Skill = 保险费**：加 skill 后多 0.5-0.9 轮验证步骤，但其稳定前缀把缓存命中推高 ~50%，在 mini 上成本完全对冲
2. **mini 思考更多**：mini 的 reasoning token 是旗舰 3 倍（132 vs 54）——能力差距部分表现为推理预算差距
3. **Context 清理端到端验证**：task 06 把预算压到 6K，清理反复触发，8/8 完成

### 9.4 方法论局限（自知）

- n=2/格太小，无统计显著性
- 任务同分布（全是 CSV/表格数据）
- scorer 只验结果不验过程

---

## 10. 测试覆盖

```bash
pytest tests/   # 40 个测试，全离线（MockProvider / 构造数据），不花钱
```

| 测试文件 | 覆盖内容 |
|---------|---------|
| `test_protocol.py` | 并行 tool_call 拼装、坏 JSON 应答、未知工具名、危险操作拒绝、length 处理、resume 消息重建 |
| `test_calculator.py` | 基本算术、数学函数、注入拒绝（`__import__`）、指数溢出、语法错误恢复 |
| `test_files_sandbox.py` | 写读回环、路径逃逸（../、符号链接、绝对路径）、错误消息可操作性、分页 |
| `test_bash_sandbox.py` | 超时击杀、危险模式匹配、Windows taskkill |
| `test_retry.py` | 429/5xx 重试、4xx 不重试、指数退避区间验证、Retry-After 头解析、连接错误、max_retries 耗尽 |
| `test_context_and_cost.py` | compact 策略触发/不触发、长上下文加价倍率、未知模型标记、Usage 累加 |

**MockProvider**（tests/conftest.py）：脚本化响应 + 记录收到的消息，让完整 agent loop 不花钱可测试。

---

## 11. Viewer（viewer/index.html）

零依赖的单文件 HTML/JS 应用（1575 行），加载 trajectory.jsonl 后可视化展示：

- 每轮的模型输入/输出
- 工具调用参数和返回值
- 四色 token 分布（prompt / cached / completion / reasoning）
- 重试事件
- Context 清理标记
- 成本累计曲线

启动方式：

```bash
python main.py serve
# 访问 http://localhost:8765/viewer/index.html?file=/runs/<run_id>/trajectory.jsonl
```

---

## 12. 依赖

```
openai>=1.60     # API 客户端
rich>=13.7       # TUI/Markdown/文本渲染
textual>=0.86    # 全屏 terminal UI
pytest>=8.0      # 仅测试用
```

没有 LangChain、没有 python-dotenv、没有 tiktoken、没有 requests。`rich/textual` 只负责 TUI 渲染，不接管 agent loop、工具协议或模型调用。

---

## 13. CH03 ProMax 增量：Coding 工具体系

CH03 的第二大步在原有 `calculator/read_file/write_file/list_files/bash` 之上，补齐 coding agent 常用的“找、读、改、验”闭环：

| 工具 | 定位 | 关键约束 |
|---|---|---|
| `edit_file` | 精确编辑已有文本文件 | 必须先 `read_file`；默认 `old_string` 唯一；多处命中需扩大上下文或 `replace_all=true` |
| `write_file` | 创建/覆写完整文件 | 创建新文件可直接写；覆写已有文件前必须先读且文件未变化 |
| `glob_files` | 按文件名找文件 | 优先 `rg --files`，稳定排序，结果限量 |
| `grep` | 按内容定位代码 | 返回文件、行号、匹配行，支持 include glob 和上下文行 |
| `file_info` | 文件体检 | 返回类型、大小、mtime、UTF-8 可读性、行数 |
| `show_diff` | 修改复核 | git 仓库走 `git diff`，非 git 场景用运行时 file history |

工具生命周期也进一步显式化：trajectory/TUI/Viewer 都能看到 `tool_queued`、`tool_validate`、`tool_permission`、`tool_start`、`tool_result`、`tool_result_persisted`、`tool_context_modified`、`tool_end`。这对应 Claude Code 类实现里的“为什么等待、为什么拒绝、是否并发、结果是否落盘、上下文状态改了什么”。

新增 coding eval：

```powershell
python eval/run_eval.py --tasks 07,08,09 --runs 1 --max-cost 0.5
```

- `07_precise_edit`：修一个函数的除零 bug，验 `read_file -> edit_file -> bash/test`，禁止整文件覆写。
- `08_search_then_patch`：跨 30 个文件定位唯一调用点，验必须用 `grep` 或 `glob_files`，且只编辑目标文件。
- `09_large_output_recovery`：故意产生超长工具输出，验 `tool_result_persisted` 后仍能根据落盘内容完成报告。

离线测试当前覆盖：

```powershell
python -m pytest -q
# 62 passed
```

---

## 14. TUI ProMax：OpenCode / Claude Code 风格交互层

`harness/tui_textual.py` 是当前默认 TUI，目标不是复刻一个 dashboard，而是做 terminal-first coding-agent REPL：

- transcript 为主屏，用户消息、assistant 回复、thinking、Build 状态在同一时间线里展示。
- `/` inline command palette 在输入框上方实时过滤命令，不打断主界面。
- 一轮用户请求对应一个 `BuildActivity`，工具调用按 `tool_call_id` 折叠进 `ToolActivity`，主屏只显示一个持续刷新的 Build 块。
- `Ctrl+O` 打开 `BuildDetailScreen`：左侧工具列表支持 hover/click/highlight，右侧显示 input/output/permission/persisted/contextModifier/lifecycle，底部提供 `Parent/Prev/Next` 导航。
- `assistant_delta.reasoning_content` 进入 `Thinking:` 渲染；没有 reasoning 的模型不会伪造思考内容。
- transcript 使用稳定 `TranscriptBody.update(...)`，不在刷新时拆掉消息 widget，避免 Textual 鼠标滚动/选择路径崩溃。
- 启动 logo 在 `harness/tui_textual.py::_tiny_agent_logo()`，是普通 Rich `Text`，可以直接替换 ASCII art 和颜色。

这层只消费 loop 已经产出的事件，不改变 agent 协议：

```text
assistant_delta.content
assistant_delta.reasoning_content
tool_call / tool_queued / tool_validate / tool_permission
tool_start / tool_result / tool_result_persisted
tool_context_modified / tool_end
```

因此 Viewer、eval、trajectory 的语义保持一致，TUI 只是把同一批事件用更适合 coding-agent 的方式折叠展示。
