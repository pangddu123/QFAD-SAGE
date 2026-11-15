# QFAD 原生记忆与 SAGE 奖励

本目录是 Qwen27B Writer–Reader 交替优化的入口。抽取、构图、事实筛选和流扩散沿用 `src/passage_entity`；奖励对应 [SAGE §4.1、Appendix O.1](https://arxiv.org/html/2605.12061v2)。

## Memory 与策略

Writer 对每个原始 passage 执行两次原生提示：NER，然后 `passage + named_entities → triples`。两次生成都是 Writer 策略动作，轨迹回报作用于它们的生成 token。提示中只有当前 passage、固定示例和本次 NER 输出。来源由环境绑定，模型不生成来源 ID。原始文本、三元组和来源索引分别保存。

`KGBuilder` 构建实体节点、passage 节点、事实边、实体–passage 边和同义边。事实存于事实向量库，不额外建立事实节点。每个检查点只用该患者当时可见的文本重建图。

Reader 执行原生流程：query 向量匹配事实 → Qwen 筛选候选事实 → 原生实体及 passage 种子权重 → Query-Aware Flow Diffusion → 返回前 5 个完整 passage。训练动作就是原生 `FactReranker` 的输出，沿用它的提示、解析、候选匹配与空筛选回退。Flow Diffusion 参数使用 `PassageEntityConfig` 默认值。

Qwen27B 底座冻结，Writer 和 Reader 各有一套 LoRA 与优化器。回答时禁用两套 LoRA，使用同一个冻结底座和原生 QA 提示。裁判在本地调用配置的 Sol 模型，只给出证据可推导性判断。

## 奖励

令 (P_k) 为实际交给回答模型的完整 passage ID 集合，(D^+) 为独立标注的支持文档集合：

\[
r_{rec}=|P_k\cap D^+|/|D^+|,\qquad
r_{pre}=|P_k\cap D^+|/|P_k|.
\]

空检索的精确率为 0；缺少支持文档标注的题目不能计算这两项。

`deducibility` 是裁判根据问题、参考答案和实际检索原文返回的布尔值；裁判不查看模型预测来代替证据判断。`answer_f1` 是冻结回答模型的预测与答案别名的最大 token F1。参考答案只进入评分端。

\[
r_{task}=\frac{\alpha r_{rec}+\beta r_{pre}+\gamma r_{ded}+\delta r_{ans}}
{\alpha+\beta+\gamma+\delta},\qquad
R_W=r_{task}-\lambda_{rep}\rho_{rep}+\lambda_{fmt}\sum_t r_t^{fmt}.
\]

重复率在原始写入序列上、图去重之前计算。每个成功解析的 passage 三元组动作获得一个单位格式分。JSON 修复后，格式错误的单条三元组被过滤，合法条目继续保留；整个动作无法解析时按 Appendix O.1 终止轨迹、回报为 0。没有原文支持率门控或整批事实语义拒绝。

六个系数由配置显式提供，不将未核实的数值称为论文默认值。每条轨迹保存六个奖励分量与最终回报。Reader 阶段使用 (R_R=r_{task})：固定图上的重复与写入格式项是组内常量，不影响 Reader 的组相对优势。

## 交替闭环

每轮执行：

1. 固定 Reader (\phi_t) 和回答底座。对同一个可见语料及训练问题采样多条 Writer 轨迹，用各自生成的图检索并计算 (R_W)，通过 GRPO 更新 (\theta_t\to\theta_{t+1})。Query 只进入检索与奖励，不进入原生 Writer 提示。
2. 冻结更新后的 Writer，用确定性解码重新抽取语料并构图。训练中得分最高的候选图不会直接成为下一阶段的 Memory。
3. 固定 Writer (\theta_{t+1})、新图和回答底座。对同一问题采样原生事实筛选动作，经相同 Flow Diffusion 获取证据，按 (R_R) 更新 Reader (\phi_t\to\phi_{t+1})。
4. 在同一批新图上评估 Reader 更新前后，保存两套适配器、优化器和各自步数。下一轮 Writer 评分使用 (\phi_{t+1})。

GRPO 的优势为组内回报减均值后除以总体标准差。等回报组不更新。损失只覆盖生成 token，按每条轨迹的输出 token 总数归一化，使用裁剪策略比率与冻结底座 KL。多次优化时旧策略 log-prob 保持固定；每个动作携带角色与采样步数，禁止跨角色或过期轨迹更新。

这实现了 **QFAD 的可学习事实筛选＋SAGE Writer 奖励的交替闭环**。SAGE 原文使用 GFM Reader；本实现保留 QFAD Reader，因此 Reader 的训练参数与论文 GFM 不同。

## 输入与执行

从 QFAD-SAGE 仓库根目录设置模块路径。以下命令沿用同级 MedMemoryBench 环境、工作区的评分脚本和已准备的医学数据。

```bash
export PYTHONPATH="$PWD/experiments:$PWD/src:$PWD/../../baselines/QAFD-RAG/.deps"
```

数据文件包含 `documents` 与 `queries`：

- document：`id, patient, stage, text`；text 是完整原始 passage。同一患者相同文本使用同一 ID。
- query：`id, patient, stage, question, answers, gold_source_ids`；可带 `answer_question`，用于保留基准的答题格式提示。

仅导出与验证数据，不加载模型或请求 API：

```bash
../MedMemoryBench/.venv/bin/python -m sage_qfad.prepare_p12 --out results/sage_qfad_native/p12_input.json
../MedMemoryBench/.venv/bin/python -m sage_qfad.run validate --dataset results/sage_qfad_native/p12_input.json
```

训练入口为 `python -m sage_qfad.run train --config CONFIG.json`，配置必须包含：

| 字段 | 内容 |
|---|---|
| `dataset`, `out` | 数据文件与新的运行目录 |
| `rounds`, `group_size` | 交替轮数与每组轨迹数，组大小至少 2 |
| `reward_weights` | `recall, precision, deducibility, answer_f1, repetition, format` 六个非负系数，前四项之和大于 0 |
| `worker_command` | 启动 `python -m sage_qfad.worker --model MODEL --out WORKER_OUT` 的 argv 数组；支持 SSH 命令数组 |
| `embedding_url`, `embedding_model` | 固定向量服务，当前为 Qwen3-Embedding-4B |
| `judge_model` | 本地裁判模型 ID |
| `seed` | 可选随机种子，默认 17 |

Worker 使用同一 Qwen3.8-27B 权重，NF4、LoRA rank 8 / alpha 16；GRPO 默认学习率 `1e-5`、clip `0.2`、KL 系数 `0.02`、每组一次优化，可通过 Worker CLI 设置。这些是本地运行超参数，非声称复现的 SAGE 超参数。`--resume` 加载两套适配器与优化器，作为新运行的初始状态。

Worker 运行在 GPU 主机时，只需本目录的 Python 包与模型依赖；原始数据处理、QFAD、向量 API 调用和裁判留在本地。GPU 选择通过启动命令的 `CUDA_VISIBLE_DEVICES` 显式指定；启动前按项目资源约定检查容量并保留 Qwen32B 服务。裁判密钥只从本地进程环境读取。

## 检查

```bash
CUDA_VISIBLE_DEVICES='' ../MedMemoryBench/.venv/bin/python -m unittest \
  sage_qfad.test_rewards sage_qfad.test_native sage_qfad.test_loop sage_qfad.test_worker -v
```

测试覆盖奖励算术、来源标签与可见性、原生图及检索结果一致性、单事实/单 passage、空事实图、示例提示顺序、两轮交替与重建、标签隔离、token 对齐、裁剪梯度、角色冻结和 checkpoint 恢复。模型更新测试仅使用 CPU 小张量夹具。

## 当前限制

- 前两位患者的已有人工原文锚点只覆盖 5 道题。导出文件保留其余未标注题目清单；全量 SAGE 检索奖励需要补足支持文档标注。已有锚点代表核验过的证据集合，不保证穷尽所有同样正确的原文出处。
- 奖励公式已实现，论文各数值系数未确认为可直接复用的配置；生产训练配置尚未选定。
- 保留 QFAD 原生文本规范化与有限步扩散。默认流扩散可达到 500 次上限，不能将其称为已经收敛；医学数值、符号、时间与断言的专门处理尚未加入。
- 离线检查不等于 Qwen27B GPU 训练与收益验证。此次未启动模型训练、全患者生成或在线裁判实验。
