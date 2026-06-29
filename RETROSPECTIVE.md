# tiny-harness 全程复盘报告

> 写给未来的自己：这份文档记录这个项目**怎么来的、路上发生了什么、为什么长这样、
> 还能往哪走**。正式设计辩护见 DESIGN.md，机器可续接的状态见 PROGRESS.md。
> 日期：2026-06-12（单日完成全部实现与验证）

---

## 0. 一页总览

**任务**：套磁考核——不用任何 agent/harness 框架，纯 Python 实现 agent loop、
≥3 个工具、context 管理、infra 与可观测性，用"读 CSV 算均值写文件"验证。

**最终成果**：
- ~2300 行 Python（核心 ~1000 + 测试 ~600 + eval ~700）+ 1575 行零依赖 viewer
- pytest **40/40**（全离线，不花钱）
- 真实 API（gpt-5.5 / gpt-5.4-mini，经中转网关）**2×2 矩阵 48/48 全过**，总开销 ~$1.4
- 核心实验发现：在这套 harness 上，**gpt-5.4-mini 以 1/6.5 成本追平 gpt-5.5 的
  100% 成功率**（天花板效应及其解读见 §4）

**立场一句话**：loop 极简（~80 行），复杂度全部投资在 harness——这个配比就是
对"harness 工程"命题的回答。

## 1. 设计期：核心决策与为什么

| 决策 | 备选方案 | 为什么这么选 | 出处/依据 |
|---|---|---|---|
| loop 只认 finish_reason，全分支处理 | 只判断"有没有 tool_calls" | `length` 不是完成、`max_tokens` 截断要 nudge 再判，漏分支是最常见协议错误 | OpenAI API 文档 |
| 工具错误回传（"ERROR: "前缀）而非终止 | 异常直接 raise | agent 的本质是靠环境反馈自纠；OpenAI 无 is_error 字段，前缀给无歧义信号 | Anthropic *Building Effective Agents* |
| 每个 tool_call_id 必须应答（含报错/拒绝/坏 JSON） | 失败的调用跳过 | 漏答任何一个 → 下次请求 400。这是协议铁律，专门写了测试 | OpenAI function calling 文档 |
| 三熔断：max_turns + max_cost + Ctrl-C 完整收尾 | 只有 max_turns | 成本是独立的失控维度；中断也要写 run_end，轨迹永远完整 | mini-swe-agent 双限制 |
| strict: true 全量启用，registry 自动补全 schema | 裸 JSON Schema | API 层面消灭参数幻觉，一行配置消掉一类错误 | OpenAI structured outputs |
| 工具结果统一截断 20K 字符，截断文本含"导航提示" | 不截断/硬截断 | 一条 cat 大文件就炸上下文；截断信息本身要告诉模型怎么缩小请求 | Claude Code 25K token 实践 |
| calculator 用 ast 白名单 | eval() + 黑名单 | eval 沙箱化是被反复证伪的方向；白名单从根上只允许算术结构 | 安全常识，有注入测试 |
| 路径校验在 resolve() 后的绝对路径上做 | 字符串查 ".." | symlink/绝对路径/盘符都能绕过字符串过滤 | — |
| bash 沙箱=超时杀进程树+cwd 锁定+危险模式 hook，**文档明示边界** | 上 Docker | 威胁模型是"模型犯傻"不是"模型作恶"；展示认知边界比堆设施更值钱 | Anthropic sandboxing 博客 |
| context 策略：清旧工具结果，不做摘要 | LLM 摘要压缩 | 清除无失真、不付额外调用、不破缓存；预算 240K 卡在 272K 加价线下是经济学硬依据 | Anthropic context editing 默认策略 |
| 触发用 API 真实 prompt_tokens，不用本地估算 | tiktoken 本地计数 | 经中转后本地 tokenizer 对不上；账单数字才是 ground truth | — |
| JSONL 事件流，llm_request 记完整 messages | 只记摘要日志 | 可复现 = 记录而非控制（temperature=0 在云端不保证确定性）；完整请求是唯一重放凭据 | Thinking Machines *Defeating Nondeterminism* |
| --replay 离线重放做成一等公民 | 只留日志 | 一个开关同时解决离线测试、零成本演示、复现验证三件事 | METR Vivaria / inspect_ai 思路 |
| provider 抽象层（loop 不碰 SDK 类型） | 直接在 loop 里调 SDK | 换协议=新增一个文件；后来 DeepSeek 方言恰好证明了这层抽象的价值 | terminal-bench 用 LiteLLM 的思路 |
| 自研重试关掉 SDK 内置 | 用 SDK max_retries | 每次退避要写进 trajectory——可观测性优先 | — |
| eval 用子进程跑 agent | 进程内 import 调用 | 环境隔离 + 熔断兜底；eval 直接消费 harness 自己的 summary.json，闭环自洽 | inspect_ai 三元组 |
| 刻意不做：sub-agent / MCP / 摘要压缩 / Docker | 全堆上 | 克制也是答题；每项在 DESIGN.md 用两句话说明"是什么+为什么不做" | — |

## 2. 实现期：值得记住的工程细节

- **离线协议测试是性价比之王**：MockProvider（脚本化响应+记录收到的消息）让
  并行 tool_call 拼装、坏 JSON 应答、429→成功的退避序列全部不花钱可验证。
  40 个测试一次全绿后，真实 API 阶段没有出过一个协议错误。
- **故障注入直接构造 httpx.Response + openai.APIStatusError**，断言退避区间
  （2^n + U(0,1)）而非具体值。
- **viewer 交给后台 agent 并行开发**，唯一契约是 telemetry.py 头注释的事件
  schema——接口先行，双线零冲突。
- **Windows 特有坑**：进程树要 `taskkill /F /T`（POSIX 是 killpg）；GBK 控制台
  会被 ▶/✅ 炸死，所有 CLI 入口统一 `sys.stdout.reconfigure(encoding="utf-8")`。

## 3. 真实 API 期：踩坑实录（每个都是面谈素材）

| # | 现象 | 根因 | 修复 | 学到什么 |
|---|---|---|---|---|
| 1 | 连不上 | .env 的 base_url 缺 `/v1`，SDK 拼出错误路径 | 补 /v1 | 中转配置的第一检查项 |
| 2 | 403「令牌额度不足本次预扣费」 | 中转按 max output 预扣，key 设了限额 | 用户提额；期间**实测 --resume 断点续跑**成功 | harness 表现完美：403 不重试、轨迹完整；模型醒来第一件事是用 calculator 验算断点前的结果——resume 的消息重建是对的 |
| 3 | DeepSeek 400「reasoning_content must be passed back」 | 思考模型方言：reasoning_content 要逐轮回传 | ModelTurn 增加字段全链路保真 | 厂商方言是 provider 层的职责 |
| 4 | 修完 #3 还是 400（第 7 轮才炸） | **没思考的轮次**上游不返回该字段，但回传时必须补空串 | provider 方言开关：检测到思考方言后，历史 assistant 消息缺字段的补 "" | 修 bug 要修到"为什么第 7 轮才炸"水落石出 |
| 5 | 523 InternalServerError 直接放弃 | 可重试集合是枚举 {429,500,502,503,504}，没料到 Cloudflare 52x | 改为 429 + 全部 5xx | 经中转的现实世界比 OpenAI 文档的错误码表更野 |
| 6 | eval 启动即崩 UnicodeEncodeError | GBK 控制台 vs ▶ 字符 | 统一 UTF-8 reconfigure | 见 §2；崩在第一行 print，一分钱没烧 |

**插曲的价值**：坑 2-5 全部发生在"协议测试全绿"之后——离线测试保证你写的协议
是对的，但**真实世界的网关方言只有真跑才能暴露**。两层验证缺一不可，这个结论
本身值得写进任何 harness 的方法论。

## 4. 实验期：2×2 矩阵结果与诚实解读

（完整数据：eval/results/20260612-120314/report.md，表格已入 DESIGN.md §8）

- **48/48 零失败 → 天花板效应**：本任务集差分不出 skill 增益。诚实承认：
  "便宜模型+经验≈旗舰"的假设**未被检验**（不是被证实）。
- **但天花板给出更强结论**：harness 质量主导该任务类——mini 以 1/6.5 成本
  （$0.0066 vs $0.0432）追平旗舰，含脏数据判断与文件名自恢复全对。
- **三个意外观测**：skill 是保险费（多 0.5-0.9 轮验证）但其稳定前缀把缓存命中
  推高 ~50%，在 mini 上成本完全对冲；mini 的 reasoning token 是旗舰 3 倍
  （132 vs 54）——能力差距部分表现为推理预算差距；上下文预算压到 6K、
  清理反复触发，8/8 完成——清理策略拿到端到端证明。
- **方法论缺口（自知）**：n=2/格太小，无统计显著性；任务同分布（全是表格数据）；
  scorer 只验结果不验过程。都是 future work 的入口。

## 5. 后续扩展路线图（按"先打破天花板，再谈花活"排序）

### 5.1 直接加分项（考核维度，各 0.5-2 天）

1. **更难的 eval 任务集**（最优先）：多文件 join、需要中间状态的迭代计算、
   故意含歧义需澄清的任务、对抗性数据（表头骗人）。目标是让裸 mini 掉到
   60-80%，skill 的增益才能差分出来——你的核心假设才有检验场。
2. **pass@k 与重复数**：每格 n≥5，报告置信区间；加 `--seed-tasks` 让 gen
   产出多套数据防过拟合。
3. **Responses API provider**：gpt-5 系正统路径，保留 reasoning items 跨轮——
   对照 Chat Completions 测 reasoning token 是否显著下降（预期下降，因为不用
   每轮重推理）。一个 provider 文件 + 一组对照数据，含金量高。
4. **Anthropic provider**：~100 行，把 DESIGN.md 里的协议映射表变成可运行代码，
   provider 抽象的完整证明。
5. **摘要式 compaction 第二层**：清理不够时升级为 LLM 摘要，正好用 mini 当
   摘要器（便宜）；与纯清理做对照实验。
6. eval 并发 runner（ThreadPool 跑子进程，矩阵 25 分钟 → 5 分钟）；
   viewer 双 run 对比视图（同任务两次运行 diff——可复现性的可视化）。

### 5.2 研究向（读研后的正经课题，和贺老师方向对口）

7. **模型级联路由（FrugalGPT 式）**：mini 先跑 → 程序化验证失败才升级 gpt-5.5。
   本实验已证明 mini 在易任务上 100%，级联的期望成本逼近纯 mini、期望质量逼近
   纯旗舰——这就是**服务计算的 QoS-成本权衡**，可以建模、可以发论文。
8. **棘轮自动化**：从失败 trajectory 自动抽取规则写回 skill 文件（agent 改进
   agent 的 harness）。每条规则可追溯到具体失败事件，闭环可量化。
9. **边缘场景实例化**：把"便宜模型"换成本地小模型（Qwen 系），harness 不变，
   测"边缘侧小模型 + 领域 harness vs 云端旗舰"的延迟/成本/质量三角——
   直接落在边缘计算 + 服务治理的叙事上。
10. trajectory 即训练数据：线性历史天然是 SFT/RL 格式（mini-swe-agent 的设计
    初衷），攒够失败样本可做针对性微调。

### 5.3 不建议做的

- MCP / 多 agent 编排 / 通用记忆系统：与"无框架理解机制"的考核目标相反，
  且在单任务场景无收益。想清楚再加，每个功能都要回答"它检验什么假设"。

## 6. 给未来自己的检查清单（面谈前过一遍）

- [ ] 能脱稿讲 §1 表格里任意一行的"为什么"和"备选为什么不行"
- [ ] 能现场演示：跑原题 → serve 打开 viewer → 指出 retry/清理标记/四色 token
- [ ] 能讲 §3 的 6 个坑，尤其 #4（为什么第 7 轮才炸）和 #2（resume 实测）
- [ ] 能诚实说出 §4 的天花板效应和 n=2 的局限，再接 §5.1 的改进方案
- [ ] 被问"为什么不用 LangChain"时：不是不会用，是这题考的就是框架底下的东西
  （然后举 tool_call_id 必须全应答这种细节）
