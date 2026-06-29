# 项目进度（供跨对话续接）

> 给新对话的 Claude：这是一个无框架 Python agent harness（套磁考核项目）。
> 设计决策与权衡全在 DESIGN.md；本文件记录完成状态与下一步。

最后更新：2026-06-12（首轮实现完成）

## 状态：全部完成 ✅（含真实 API 验证 + 2x2 矩阵实验 48/48 全过）

**矩阵实验结论已写入 DESIGN.md §8**：四个格子全部 12/12，gpt-5.4-mini 以
1/6.5 成本追平 gpt-5.5（天花板效应：本任务集差分不出 skill 增益，诚实解读
见 DESIGN.md）。完整数据在 eval/results/20260612-120314/report.md。
本次真实 API 总开销约 $1.4。

### 真实 API 阶段实录（2026-06-12，中转站 api.gpt.ge）

- gpt-5.5 原题：**7 轮 23s completed**，答案正确，$0.0434，缓存命中 1024 tok
- gpt-5.4-mini 原题：**4 轮 12s completed**，$0.0039（约 1/11 成本）
- 路上发现并修复的真实世界问题（全部有对应代码注释，面谈可讲）：
  1. base_url 缺 `/v1`（SDK 拼路径失败）
  2. 中转预扣费 403（令牌限额）——harness 正确表现：不重试、轨迹完整、
     `--resume` 断点续跑实测成功（模型还主动用 calculator 交叉验证了断点前的结果）
  3. **DeepSeek 思考方言**：thinking 模型要求 reasoning_content 逐轮回传，且
     "没思考的轮次"也要补空串——provider 层方言开关解决，不污染标准协议路径
  4. 中转网关吐 Cloudflare 52x（523）——可重试集合从枚举改为 429 + 全部 5xx

| 模块 | 状态 | 验证方式 |
|---|---|---|
| harness/（config/telemetry/providers/tools/context/loop/hooks/skills） | ✅ | pytest 40 passed |
| main.py CLI（run / --replay / --resume / serve） | ✅ | 离线端到端：手工轨迹 --replay 跑通全链路，工具真实执行，均值与 scorer 独立计算一致 |
| tests/（单元+协议 mock+故障注入） | ✅ 40 passed | `python -m pytest tests/ -q` |
| eval/（6 任务 + scorer + 2x2 矩阵 runner） | ✅ | 全部 gen/scorer 离线验证 + 负样本判负 |
| skills/csv-data-processing.md | ✅ | 7 条棘轮式经验规则 |
| viewer/index.html（1575 行零依赖） | ✅ | 后台 agent 自检：无头 Edge 渲染、10k 行 962ms、9 种事件全覆盖、XSS 转义；内置"加载演示数据"按钮 |
| DESIGN.md / README.md | ✅ | |
| **真实 API 端到端** | ⬜ **下一步** | 见下方清单 |

代码规模：Python ~2300 行（含测试）+ viewer 1575 行。

## 下一步（用户填好 .env 后按序执行）

1. `cp .env.example .env`，填 OPENAI_API_KEY、OPENAI_BASE_URL（中转，含 /v1）、
   TINY_HARNESS_CHEAP_MODEL（中转实际有的便宜模型名）
2. 冒烟：`python demo.py && python main.py "读 data.csv，算第三列的均值，写到 mean.txt" --workdir ./workspace`
   - 若中转不支持 strict/某参数报 400：看报错信息，candidates 是去掉 strict 字段
     （registry.py `openai_tool_schemas`）或不传 reasoning_effort（默认就不传）
   - 若 usage 的 *_details 缺失：已防御（按 0 计），cached/reasoning 显示 0 属正常
3. 看轨迹：`python main.py serve` → viewer 里检查 reasoning token 与缓存命中是否有值
4. 小规模 eval：`python eval/run_eval.py --tasks 01,02 --runs 1`（约 2 次运行，验证管线）
5. 全量：`python eval/run_eval.py --runs 3`，然后 2x2 实验：
   `python eval/run_eval.py --matrix --runs 3`（注意成本 ≈ 任务数6 x 组合4 x 3 次，
   先用 --max-cost 0.5 兜底；report.md 自动生成）
6. 把 report.md 的 2x2 表格写进 DESIGN.md §8 的"实验结果"小节（目前只有假设与方法）

## 关键约定（改动需同步处）

- 事件 schema：harness/telemetry.py 头注释 = viewer + tests 的三方契约
- 工具错误信号：tool 消息 content 前缀 `"ERROR: "`（OpenAI 无 is_error 字段）
- 终止原因枚举：completed / max_turns / max_cost / truncated / interrupted / error
- 协议铁律：每个 tool_call_id 必须有应答（含报错/拒绝/非法 JSON 三种情形）
- demo-replay：runs/demo-replay/ 是手工构造的演示轨迹，可随时
  `python main.py --replay demo-replay --workdir ./workspace` 零成本复现

## 已知限制（DESIGN.md §4/§9 有完整讨论，面谈可作答）

- 沙箱是事故防护级（threat model = 模型犯傻），对抗级需 OS 隔离/容器，接口已预留
- Chat Completions 不保留 reasoning items（中转兼容性取舍），reasoning 占比会偏高
- viewer 实时模式整文件轮询（渲染增量、fetch 非增量）
