# tiny-harness

无框架（no LangChain / inspect_ai）的 Python agent + 生产级 harness。
Loop 本体 ~80 行；功夫全在 harness：协议正确性、重试、沙箱、上下文管理、
JSONL trajectory、成本台账、离线重放、eval、可视化。设计决策与权衡见
[DESIGN.md](DESIGN.md)，全程复盘/踩坑实录/扩展路线见
[RETROSPECTIVE.md](RETROSPECTIVE.md)，进度见 [PROGRESS.md](PROGRESS.md)。

## 快速开始

```bash
pip install -r requirements.txt
cp .env.example .env          # 填 OPENAI_API_KEY 和中转 OPENAI_BASE_URL（含 /v1）

# 生成演示数据并跑题目原题
python demo.py                # 在 ./workspace 生成 data.csv
python main.py "读 data.csv，算第三列的均值，写到 mean.txt" --workdir ./workspace
```

运行结束输出：终止原因、轮数、四类 token（input/cached/output/reasoning）、
美元成本、final answer，以及 trajectory 路径。

## 看轨迹（可视化）

```bash
python main.py serve
# 浏览器打开 http://localhost:8765/viewer/index.html?file=/runs/<run_id>/trajectory.jsonl
# 或直接打开 viewer/index.html 拖入 jsonl 文件；页内有"加载演示数据"按钮
```

时间线展示每轮的思考/工具调用/结果（错误红框）、retry 与 context 清理标记；
仪表盘有每轮 token 堆叠图（含 reasoning 占比）、累计成本、上下文水位线。

## 重放与续跑（可复现性）

```bash
python main.py --replay <run_id>            # 离线重放：不打 API、零成本、确定性
python main.py --resume <run_id> "继续：再算第四列"   # 从历史消息现场继续
```

## 测试（全部离线，不消耗 API）

```bash
python -m pytest tests/ -q    # 40 项：单元 + 协议(mock) + 故障注入(429/400/退避)
```

## Eval

```bash
python eval/run_eval.py --runs 3                  # 6 任务 x 3 次
python eval/run_eval.py --tasks 02 --skill csv-data-processing
python eval/run_eval.py --matrix --runs 3         # {主模型,便宜模型} x {裸,skill} 2x2 实验
```

输出 `eval/results/<时间戳>/report.md`：成功率 / 平均轮数 / 平均成本 /
reasoning token / 缓存命中的汇总表。

## 常用参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--max-turns` | 30 | 轮数熔断 |
| `--max-cost` | 1.0 | 美元成本熔断 |
| `--context-budget` | 240000 | input token 预算，超过触发工具结果清理（避开 272K 加价线） |
| `--reasoning-effort` | 服务端默认 | none/low/medium/high/xhigh |
| `--skill` | - | 注入领域经验包（skills/*.md），可重复 |
| `--yolo` | off | 跳过危险命令确认（eval 自动化用） |

## 目录

```
harness/            核心：loop / providers(重试+replay) / tools(沙箱) / context / telemetry / hooks / skills
tests/              40 项离线测试（协议正确性是重点）
eval/               6 任务 + scorer + 2x2 矩阵 runner
skills/             领域经验包（便宜模型增强实验用）
viewer/index.html   零依赖轨迹可视化
runs/<run_id>/      每次运行的 trajectory.jsonl + summary.json
```
