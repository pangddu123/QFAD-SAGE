# SAGE 开发交接与实验环境

记录日期 2026-11-15，核对的算法提交为 `2f18839`。本仓库默认开发分支是 `main`。开发机编辑、提交代码，原实验环境负责读取医学数据、完整测试和运行 Writer–Reader 交替训练。

## 在新位置开发

```bash
git clone --branch main https://github.com/pangddu123/QFAD-SAGE.git
cd QFAD-SAGE
git branch --show-current
python3 -m compileall -q experiments/sage_qfad
PYTHONPATH=experiments python3 -m unittest sage_qfad.test_rewards
PYTHONPATH=experiments python3 -m sage_qfad.run --help
```

这些命令只需要 Python 标准库，覆盖语法、奖励计算和 CLI 加载。若在另一个 Git 远端开发，把选定提交同步到实验仓库的 `main`，记录提交号。在原实验机的 `QFAD-SAGE/` 执行 `git pull --ff-only origin main`，用 `git rev-parse HEAD` 对照后再验证。两个远端之间需要显式同步。

| 要改的行为 | 主要代码 |
|---|---|
| 医学输入转换、时间可见性与来源标注 | `experiments/sage_qfad/prepare_p12.py`、`data.py` |
| QFAD 原生写入、检索、重建图 | `experiments/sage_qfad/native.py`、`src/passage_entity/` |
| 证据可推导性 Prompt、回答评分与 SAGE 奖励 | `experiments/sage_qfad/evaluation.py`、`rewards.py` |
| Writer / Reader 交替顺序、策略通信、GRPO 与适配器 | `experiments/sage_qfad/loop.py`、`policy.py`、`worker.py` |

## 原实验环境

控制端工作区目录名为 `medicalMemory/`，仓库目录为其中的 `QFAD-SAGE/`。系统为 Ubuntu 24.04.4 LTS、Linux x86_64。控制端使用 `MedMemoryBench/.venv/bin/python`（CPython 3.12.3）与工作区上一层的 `baselines/QAFD-RAG/.deps`。此次读取到的主要包版本为 torch 2.12.1+cpu、numpy 1.26.4、igraph 0.11.9、pandas 2.2.3、pyarrow 19.0.1、openai 1.109.1、httpx 0.28.1。根目录 `requirements.txt` 描述上游 QFAD 的通用依赖。

| 资源 | 位置与用途 |
|---|---|
| 医学基准 | 工作区同级项目 `MedMemoryBench/`（相对仓库为 `../MedMemoryBench/`），基点 `d9c0d5a`。现有检出含本地修改，英文混合输入在 `data/MedMemoryBench_EN_mixed/` |
| 数据导出工具 | 工作区 `scripts/qafd_medmemorybench_mixed.py` 及其 RAW-RAG 适配脚本。导出还读取工作区 `results/qafd_medmemorybench_en_mixed_gpt4omini_20261111/` 的 passage ID、`results/medical_sage/extraction_training_plan.json` 的已核验原文锚点 |
| 已导出的 SAGE 数据 | 工作区 `results/sage_qfad_native/p12_input.json`。复核为 3,594 条 document、5 道带来源标注的问题、192 道缺少该标注的问题。训练也可以读取相同 schema 的其他 JSON |
| 检索和评分服务 | 向量服务由配置 `embedding_url`、`embedding_model` 指定。当前实验沿用工作区的 Qwen3-Embedding-4B 服务。证据裁判由 `judge_model` 指定，`scripts/medmemory_scoring.py` 从实验机环境读取 `EXPERIMENT_API_BASE_URL`、`EXPERIMENT_API_KEY` |
| 策略 worker | 由运行配置中的 `worker_command` 明确指定启动命令、Python、GPU、模型和输出目录。策略在现有 A100 训练主机上加载本地 Qwen3.8-27B 权重。已有 27B 环境记录为 A100 80GB、torch 2.13.0+cu129、transformers 5.16.1、accelerate 1.15.0、peft 0.21.3、bitsandbytes 0.50.2；这些记录不代表 SAGE 已完成 GPU 训练 |
| 模型与产物 | Worker 对同一 NF4 基座分别维护 Writer、Reader LoRA，回答时禁用适配器。控制端 `out` 存每轮记录与 `complete.json`；同级 `<out>_worker/` 记录通信与 stderr；模型及优化器保存在 worker 自己的输出目录 |

Git 跟踪 QFAD 源码、SAGE 训练包、测试和方法说明。解释器、工作区适配脚本、患者输入、已导出 JSON、模型、运行配置中的实际服务信息、凭据与训练产物由原实验环境管理。运行参数字段、奖励系数和 worker 启动方式见 [SAGE 方法与配置说明](../experiments/sage_qfad/README.md)。当前分支未提供选定的生产训练配置，本次未探测在线服务。

## 在原实验机验证改动

从实验机的 `QFAD-SAGE/` 执行。第二条命令使用本地已导出的数据验证 schema，第三条执行 CPU 测试。

```bash
export PYTHONPATH="$PWD/experiments:$PWD/src:$PWD/../../baselines/QAFD-RAG/.deps"
../MedMemoryBench/.venv/bin/python -m sage_qfad.run validate --dataset ../results/sage_qfad_native/p12_input.json
CUDA_VISIBLE_DEVICES='' ../MedMemoryBench/.venv/bin/python -m unittest sage_qfad.test_rewards sage_qfad.test_native sage_qfad.test_loop sage_qfad.test_worker
```

真正的训练入口是 `python -m sage_qfad.run train --config CONFIG.json`。已验证范围包括奖励、图检索、交替循环和小张量参数更新。该范围没有给出 27B 策略的配对 QA 增益或全患者 RAW-RAG 对比结果。
