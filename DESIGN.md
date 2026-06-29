# tiny-harness 设计文档

无框架（no LangChain / inspect_ai / Agent SDK）的 Python agent 实现。
唯一运行时依赖是 OpenAI SDK——它是 API 客户端，不是 agent 框架；重试、循环、
上下文、可观测性全部自研，且 SDK 内置重试被显式关闭（见 §4）。

**核心立场：loop 应当极简，复杂度应当全部投资在 harness 上。**
依据：Anthropic *Building Effective Agents* 的简单性原则（"为需求构建正确的
系统，而非最复杂的系统"）；mini-swe-agent 用 100 行 loop 在 SWE-bench Verified
拿到 74%+ 的实证。本项目 loop.py 的循环本体约 80 行，其余 ~900 行全是 harness：
工具质量、上下文管理、可靠性、可观测性。这个配比本身就是对"harness 工程"
这一命题的回答——agent = model + harness，模型不可改，harness 是唯一杠杆。

```
main.py ──► loop.py ──────► providers/ ──► OpenAI API（或 Replay 离线重放）
   │           │  ▲              │ 重试: 429/5xx 退避+抖动；usage 四字段
   │           ▼  │              ▼
   │        tools/ registry   telemetry.py ──► runs/<run_id>/trajectory.jsonl
   │        calculator/files/bash               │            summary.json
   │           │                                ▼
   │        hooks.py（危险命令门控）          viewer/index.html（零依赖可视化）
   │        context.py（预算+清理）
   └─ eval/run_eval.py（Dataset → Solver子进程 → Scorer）
```

---

## 1. Agent loop（loop.py）

循环只认 `finish_reason`，全分支处理：

| finish_reason | 处理 | 为什么 |
|---|---|---|
| `tool_calls` | 执行全部调用 → 应答 → 继续 | 多个调用真并发（ThreadPoolExecutor），模型既然并行发起就并行执行 |
| `stop` | 以 completed 结束 | 模型自然交付即任务边界 |
| `length` | 第一次：nudge 续写并要求收敛；第二次：以 truncated 结束 | **截断不是完成**。直接当完成是最常见的协议错误 |

三重熔断：`max_turns`（防死循环）、`max_cost_usd`（防钱包失血，mini-swe-agent
同款双限制）、Ctrl-C（捕获后补写 run_end 事件再退出，trajectory 永远完整）。

**工具错误回传而非终止**：所有工具异常 catch 后以 `"ERROR: "` 前缀回传
（OpenAI 协议没有 Anthropic 的 `is_error` 字段，用前缀给无歧义信号），让模型
自纠。只有不可恢复错误（4xx 协议错误、重试耗尽）终止运行。错误文本必须
**可操作**——"文件不存在 + 目录里实际有哪些文件"，而非裸 traceback
（Anthropic *Writing Effective Tools*：错误信息是恢复设计的核心）。

**协议铁律**（tests/test_protocol.py 逐条离线验证）：assistant 消息里每个
`tool_call_id` 必须有对应 `role:"tool"` 应答——包括工具报错、参数 JSON 非法
（模型可能吐坏 JSON，解析失败本身也是要回传的错误）、被安全策略拒绝三种情形。
漏答任何一个，下次请求 400。

## 2. Provider 层（providers/）

loop 只认识自定义的 `ModelTurn`/`ToolCallRequest`，不接触厂商 SDK 类型。
换协议 = 新增一个 ~100 行的 Provider 实现。两套主流协议的映射作为换协议的施工图：

| 维度 | Anthropic Messages | OpenAI Chat Completions（本实现） |
|---|---|---|
| 停止信号 | `stop_reason`: tool_use/end_turn/max_tokens | `finish_reason`: tool_calls/stop/length |
| 工具调用 | content 中的 `tool_use` 块 | `message.tool_calls[]`，arguments 是 JSON **字符串** |
| 结果回传 | `tool_result` 块合并进一条 user 消息 | 每个 id 一条 `role:"tool"` 消息 |
| 错误标记 | `is_error: true` | 无，约定 `"ERROR: "` 前缀 |
| 缓存 | 手动 `cache_control` 断点 | 自动（≥1024 token 前缀），命中价 0.1x |
| usage | input/output + cache 两字段（互斥相加） | prompt/completion + details 子集字段 |

**为什么用 Chat Completions 而非官方推荐的 Responses API**：gpt-5.5 在
Responses API 下可跨轮保留 reasoning items（工具调用之间不丢思维链，省去
重复推理的 token），是更优解；但考核给的中转网关普遍只透传
`/v1/chat/completions`。这是部署约束下的显式取舍，不是无知——代价是每轮
工具返回后模型重新推理，reasoning token 占比会偏高（可在 viewer 中直接观察）。

**重试矩阵**（自研而非用 SDK 内置，因为每次退避要写进 trajectory）：

- 可重试：429 / 500 / 502 / 503 / 504 / 连接错误。429 优先遵循 `Retry-After` 头。
- 不可重试：400 / 401 / 403 / 404 / 413 / 422——重试只会原样重复失败。
- 退避：`min(2^attempt + U(0,1), 60)`，默认 5 次。**抖动必须有**：同时被打回的
  客户端若同步重试会复现过载（thundering herd）。

## 3. 工具层（tools/）

- **描述即接口**：每个工具描述按"给新员工写入职文档"的标准写，包含
  何时该用我/何时不该（read_file 主动把大文件让给 bash）、失败后该干什么
  （"被拒绝就换安全路径，别换措辞重试"）。Anthropic 在 SWE-bench 上仅靠精修
  工具描述就显著降错——这是 harness 里性价比最高的优化点。
- **strict: true 全量启用**：registry 自动补全 `additionalProperties: false` +
  字段全 `required`（可选语义用 `["T","null"]`）。参数合法性由 API 层保证，
  消灭一整类参数幻觉。
- **统一截断（默认 20K 字符）**：一条 `cat 大文件` 就能炸掉上下文。截断保留
  头 80% + 尾 10%，中间插入"如何缩小请求"的指引——截断信息本身也是给模型的
  导航（参照 Claude Code 单工具响应 25K token 上限的实践）。
- calculator 用 ast 白名单而非 `eval`：eval 沙箱化是被反复证伪的方向，
  白名单从根上只允许算术结构存在（测试含 `__import__`/dunder 链注入用例）。

## 4. 沙箱（bash.py + files.py）：威胁模型决定强度

本场景的威胁模型是**模型犯傻**（误删、死循环、读爆上下文），不是模型作恶。
按该模型配置三层防护，并诚实声明边界：

1. **硬约束**：`cwd` 锁 workdir、超时杀整棵进程树（Windows `taskkill /T` /
   POSIX `killpg`）、输出截断。文件工具所有路径 `resolve()` 后必须仍在
   workdir 内——在解析后的绝对路径上判定，字符串层面查 `..` 防不了
   symlink/绝对路径。
2. **软闸 + hook**：危险命令模式（rm -rf /、sudo、curl|sh、fork bomb 等）命中
   后走门控：`--yolo` 放行留痕 / 交互终端询问 / 非交互拒绝并回传理由。
3. **明示边界**：字符串过滤**不是安全边界**（base64、变量拼接、子 shell 全能
   绕过）。对抗级隔离的正确做法是 OS 原语（bubblewrap/Seatbelt，参
   anthropic-experimental/sandbox-runtime）或容器（terminal-bench/inspect_ai
   的标准配置）。本项目不伪装拥有它没有的安全性。

## 5. Context 管理（context.py）

- **触发依据是 API 返回的真实 `prompt_tokens`**，不是本地估算——本地 tokenizer
  对不上服务端（尤其过中转），账单数字才是 ground truth。
- **策略：清除旧工具结果**（替换为占位符，保留最近 3 条），而非摘要压缩。
  这是无信息再加工失真的最小干预（Anthropic context editing API 的默认策略
  同此：trigger=100K / keep=3）；摘要压缩有失真风险、要额外付一次 LLM 调用、
  且必然打破前缀缓存，作为第二层手段留在此处讨论而刻意不实现。
- **预算默认 240K 有经济学依据**：gpt-5.5 单请求 input 超 272K 后整个请求按
  input x2 / output x1.5 计费。上下文管理在这里不只是防 context rot 的软道理，
  是"超线钱翻倍"的硬闸。
- 只改写 `role:"tool"` 消息的 content，消息结构与 tool_call_id 链保持完整
  （协议要求）；占位符明示"已清除约 N token，需要可重新调工具"。

## 6. 可观测性（telemetry.py + viewer/）

- **JSONL 事件流**：一行一个自包含事件，append-only、可流式 tail、崩溃只丢
  最后一行（评测 harness 的事实标准，terminal-bench/harbor 正在 RFC 统一同款
  格式）。9 种事件类型见 telemetry.py 头注释——那是 viewer 与 tests 共同
  依赖的契约。
- **成本公式**：`(prompt−cached)·P_in + cached·P_cached + completion·P_out`。
  reasoning token 是 completion 的子集，**不重复计费但单独呈现**——
  "思考花了多少钱"是 GPT-5 系产品观测的关键维度。未知模型计 0 并显式标记
  `pricing_unknown`，绝不静默编造成本。
- **可复现 = 记录而非控制**：temperature=0 在云端推理下不保证逐位确定
  （服务端动态 batching 改变归约顺序，Thinking Machines *Defeating
  Nondeterminism* 给出根因；OpenAI 的 seed 也已走向弃用）。因此承诺的是
  完整凭据链：run_id + 全量请求/响应 + SDK 版本 + request_id。
  `--replay <run_id>` 是该承诺的可执行兑现：离线逐响应重放历史运行，
  零成本演示与回归。`--resume <run_id>` 则从消息现场继续。

## 7. Eval（eval/）

inspect_ai "Dataset → Solver → Scorer" 三元组的微缩版：任务 = 目录
（prompt + 确定性 gen + 程序化 scorer），agent 以子进程运行（隔离 + 熔断兜底），
结果从 harness 自己的 summary.json 读取——eval 消费的就是可观测性输出，闭环自洽。

| 任务 | 考点 |
|---|---|
| 01 csv_mean | 题目原题，基线 |
| 02 wrong_filename | 工具报错自恢复（list_files 自救）——题面要求"报错恢复"的端到端验证 |
| 03 dirty_data | 数据判断力：无效值应剔除而非按 0 计（scorer 能区分这两种答案） |
| 04 tool_chain | bash→计算→calculator 交叉验证→写文件的多工具协作 |
| 05 big_file | 30K 行：read_file 截断指引能否把模型导向流式方案 |
| 06 context_pressure | 预算压到 6K 强制触发清理——清理策略的端到端验证 |

## 8. 实验：便宜模型 + 领域经验 ≈ 贵模型？

`run_eval.py --matrix` 跑 {gpt-5.5, 便宜模型} × {裸 harness, +skill} 的 2×2。
skill（skills/csv-data-processing.md）是注入 system prompt 的领域专家经验，
每条规则对应一类真实失败模式（棘轮原则：每行经验都应能追溯到一次具体失败）。

假设：在验证器硬（程序化判分）、领域窄（表格数据）的任务上，便宜模型 + 经验
注入能以数分之一成本逼近旗舰模型成功率。这是"边缘侧小模型 + 领域定制 harness"
的成本-质量权衡的微缩实验——服务计算视角下，它就是 LLM 服务治理问题：重试/
超时是 QoS，token 预算是资源管理，trajectory 是服务可观测性。

### 实验结果（2026-06-12，中转网关，6 任务 x 2 重复 = 每格 12 次）

| model | skill | 成功率 | 平均轮数 | 平均成本$ | 平均 reasoning tok | 平均缓存命中 tok |
|---|---|---|---|---|---|---|
| gpt-5.4-mini | 裸 | **12/12** | 5.7 | **0.0066** | 132 | 4139 |
| gpt-5.4-mini | +skill | **12/12** | 6.1 | 0.0065 | 152 | 6144 |
| gpt-5.5 | 裸 | **12/12** | 6.3 | 0.0432 | 54 | 1664 |
| gpt-5.5 | +skill | **12/12** | 7.2 | 0.0561 | 34 | 5803 |

读数（诚实解读，不过度宣称）：

1. **48/48 零失败，出现天花板效应**：本任务集上裸 harness 的便宜模型已饱和，
   "skill 提升成功率"的假设无法在此差分检验——需要更难的任务集才能拉开
   （future work）。但这恰恰给出更强的结论：
2. **harness 质量主导这一任务类**：在协议正确、错误可恢复、工具描述到位的
   harness 上，便宜模型以 **1/6.5 的成本**（$0.0066 vs $0.0432）做到与旗舰
   完全相同的成功率，包括脏数据判断（8/8 次正确剔除 12 个无效值而非按 0 计）
   和错误文件名自恢复（8/8）。"模型不可改，harness 是杠杆"在此有了本组数据。
3. **skill 的代价可观测**：skill 增加约 0.5-0.9 轮（更多交叉验证步骤），在已
   饱和的任务上是纯保险费；但其拉长的稳定 system 前缀显著提高缓存命中
   （mini: 4139→6144），在 mini 上几乎完全对冲了额外轮次的成本（$0.0066→$0.0065）。
4. **小模型"想"得更多**：mini 平均 reasoning token（132-152）约为 gpt-5.5
   （34-54）的 3 倍——能力差距部分表现为推理预算差距，这正是 viewer 把
   reasoning 单列一色的价值。

## 9. 刻意不做的（克制也是答题）

- **摘要式 compaction**：见 §5 的三条代价；当前任务尺度用不到第二层。
- **Sub-agent**：上下文隔离收益对单任务场景为零，token 成本翻数倍。
- **MCP**：标准化外部集成与"无框架理解机制"的考核目标相反。
- **跨会话记忆**：没有多会话需求；skills 机制已覆盖"经验持久化"的本质。
- **Docker 沙箱**：威胁模型不要求（§4）；接口已预留（换 provider 同理，
  bash 执行换 `docker exec` 是一行级改动，mini-swe-agent 同款设计）。

## 10. Prior art

- Anthropic: *Building Effective Agents* / *Effective Context Engineering* /
  *Writing Effective Tools* / Claude Code sandboxing —— 原则与默认值的主要来源
- mini-swe-agent（SWE-bench 团队）—— 极简哲学与双熔断；本项目借其"线性历史 =
  trajectory = 训练数据"的思路，舍其"纯 bash 无工具协议"的激进选择（考核
  要求展示 function calling 协议能力）
- inspect_ai（UK AISI）/ terminal-bench —— eval 三元组与 JSONL 轨迹格式
- Thorsten Ball *How to Build an Agent* —— "LLM + loop + enough tokens"
- OpenAI 文档：function calling / prompt caching / reasoning（usage 字段与
  计费规则的一手依据，2026-06 核实）
