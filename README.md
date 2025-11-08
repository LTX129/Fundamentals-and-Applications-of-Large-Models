

# Transformer Summarization (Gigaword Subset)

本项目为《Fundamentals and Applications of Large Models》课程期中实验，实现了一个从零搭建的 **Encoder-Decoder Transformer**，并在 **Gigaword 标题生成任务**上进行了训练及消融实验。模型包含：

- Multi-Head Self-Attention
- Position-wise Feed Forward Network
- 残差连接 + LayerNorm
- 可选共享词嵌入
- Learned Positional Encoding
- Greedy / Beam Search 解码
- Noam (Warmup) 学习率调度
- Label Smoothing、AMP、梯度裁剪

并支持 **DUC2004 多参考 BLEU/ROUGE 评测**。

---

## 📦 环境要求

项目基于 miniconda，建议使用 GPU 训练。

| 组件 | 推荐版本 |
|---|---|
| Python | ≥ 3.9 |
| PyTorch | ≥ 2.1 (支持 CUDA) |
| CUDA | ≥ 11.7 |
| GPU | 至少 8GB 显存（16GB+ 推荐） |

创建虚拟环境示例：

```bash
git clone 
conda create -n largemodel python=3.10 -y
conda activate largemodel
cd largemodel
pip install -r requirements.txt 
```
## 📂 代码结构
```
largemodel/
├── checkpoints/                      # 训练/评测保存的权重（best.pt / last.pt 等）
├── dataset/
│   └── sumdata/
│       ├── DUC2003/
│       ├── DUC2004/                  # input.txt + task1_ref*.txt（多参考评测）
│       ├── Giga/
│       └── train/                    # train.article.txt / train.title.txt / …
│   ├── metafile.yaml
│   └── README.md
├── results/
│   ├── outputs/                      # 训练生成的曲线图、预测与中间结果（work_dir 可指向此处）
│   └── result_terminal collect.txt   # 终端日志汇总
├── scripts/
│   ├── env.sh                        # 环境变量（设置 PYTHONPATH 等）
│   ├── run.sh                        # baseline 训练脚本（含固定随机种子）
│   ├── evaluate.sh                   # DUC2004 评测脚本
│   ├── run_ablation.sh               # 一键运行三组消融实验
│   ├── plot_ablation.py              # 生成消融可视化（如 rouge1_vs_epoch.png 等）
│   └── collect_results.py            # 收集/整理实验日志与指标
├── src/
│   ├── main.py                       # 训练 / 验证 / 保存加载
│   ├── transformer.py                # Encoder-Decoder & Greedy/Beam
│   ├── layers.py                     # MHA / FFN / LayerNorm / PosEnc
│   ├── tokenizer.py                  # 词表 & 编码解码 & 特殊符号
│   ├── gigaword.py                   # 数据读取 & DUC2004 多参考评测
│   ├── schedule.py                   # Noam 学习率调度
│   ├── metering.py                   # 滑动均值 / 可视化辅助
│   └── bleu_rouge.py                 # BLEU / ROUGE（多参考）
├── README.md
└── requirements.txt
```
_

## ⚙️ 参数说明 / Arguments

本项目支持通过命令行配置模型结构、训练策略和解码方式。所有参数均在 `src/main.py` 中定义，可通过 `--flag value` 的形式修改。

### 基本运行参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--mode` | `train` / `eval` | `train` | 选择训练或评测模式 |
| `--data_dir` | `str` | 必填 | 训练数据路径（Gigaword train） |
| `--valid_dir` | `str` | `None` | 验证 / 测试数据路径（如 DUC2004） |
| `--work_dir` | `str` | `outputs/exp1` | 结果输出目录（loss 曲线 / 可视化 / 生成结果） |
| `--ckpt_dir` | `str` | `checkpoints` | 模型权重保存目录 |
| `--ckpt_path` | `str` | `checkpoints/best.pt` | 用于评测或继续训练的模型文件路径 |

---

### 模型结构参数

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--vocab_size` | `int` | `32000` | 词表大小（建议与 dataset 配套） |
| `--max_len` | `int` | `128` | 序列最大长度（文章 + 标题均会截断/填充） |
| `--d_model` | `int` | `256` | Transformer 隐层维度 |
| `--n_heads` | `int` | `4` | 多头注意力头数 |
| `--num_layers` | `int` | `4` | Encoder / Decoder 堆叠层数 |
| `--ff_dim` | `int` | `1024` | 前馈网络内部维度 |
| `--dropout` | `float` | `0.1` | Dropout 比例 |
| `--share_embeddings` | `flag` | `False` | 是否共享 source / target 词嵌入（小模型建议开启） |

---

### 训练参数

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--epochs` | `int` | `10` | 训练轮数 |
| `--batch_size` | `int` | `64` | 批大小 |
| `--grad_accum` | `int` | `1` | 梯度累积（提高等效 batch size） |
| `--lr` | `float` | `3e-4` | 初始学习率（与 Noam 调度配合使用） |
| `--warmup_steps` | `int` | `4000` | Noam warmup 步数 |
| `--weight_decay` | `float` | `0.0` | AdamW 权重衰减 |
| `--max_grad_norm` | `float` | `1.0` | 梯度裁剪上限 |
| `--amp_dtype` | `bf16` / `fp16` / `fp32` | `bf16` | 是否启用自动混合精度训练 |
| `--no_amp` | `flag` | `False` | 禁用 AMP（若想强制 FP32 训练） |
| `--no_compile` | `flag` | `False` | 禁用 `torch.compile`（避免 CUDAGraph 冲突） |

---

### 数据与词表

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--rebuild_vocab` | `flag` | `False` | 强制重新构建 `vocab.json` |
| `--min_vocab_size` | `int` | `100` | 若已存在词表小于该阈值，则自动重建 |
| `--train_limit` / `--valid_limit` | `int` | `None` | 仅用于调试（限制样本数量） |

---

### 消融实验（Ablation）

| 参数 | 说明 |
|------|------|
| `--ablate none` | 完整模型（默认） |
| `--ablate no_posenc` | 去除位置编码 |
| `--ablate single_head` | 将多头注意力降为单头 |
| `--ablate no_label_smoothing` | 关闭 label smoothing |

---

### 解码（生成）策略

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `--decode` | `greedy` / `beam` | `beam` | 解码方式 |
| `--beam_size` | `int` | `4` | Beam 宽度 |
| `--length_penalty` | `float` | `0.6` | 长度惩罚系数（越大越鼓励更长输出） |
| `--min_gen_len` | `int` | `5` | 最短生成长度（防止提前 EOS） |
| `--no_repeat_ngram_size` | `int` | `0` | 防止重复 n-gram |
| `--eos_bias` | `float` | `2.0` | 调节生成结束倾向（高 → 提前结束） |

---

### 📌 推荐用于复现实验的参数

```bash
--vocab_size 32000 --batch_size 128 --grad_accum 2 \
--d_model 192 --n_heads 3 --num_layers 2 --ff_dim 768 \
--warmup_steps 8000 --lr 5e-4 --epochs 3 --amp_dtype bf16 \
--dropout 0.1 --decode beam --beam_size 6 \
--length_penalty 1.0 --min_gen_len 6 --max_len 20 \
--no_repeat_ngram_size 3 --eos_bias 0.5 --share_embeddings
```

⸻
## 🎯 固定随机种子（可复现实验结果）
作业中没有设置随机种子，如有需要可以在 main.py 顶部加入：

```bash
import random, torch, numpy as np
def set_seed(seed=3407):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed(42)
```
⸻
## 🚀 训练（Baseline）
使用 scripts 里面的 run.sh 脚本开始训练
```bash
chmod +x scripts/run.sh
bash scripts/run.sh
```

⸻

## 🧪 DUC2004 多参考评测
训练过程自带 eval 环节，如果想单独进行 eval，可使用 scripts 里的 evaluate.sh 进行评估
```bash
chmod +x scripts/evaluate.sh

# 直接评测（用默认路径）
scripts/evaluate.sh

# 覆盖某些参数（比如换解码更短的标题）
scripts/evaluate.sh --eos_bias 0.8 --max_len 18
```
⸻
## 🔬 消融实验
可使用 scripts 里的 run_ablation.sh 进行消融实验
```bash
# 赋权
chmod +x scripts/run_ablation.sh
# 运行所有消融
bash scripts/run_ablation.sh
# 汇总结果 → CSV/Markdown
python scripts/collect_results.py /root/workspace/tmp/largemodel/outputs
# 画对比曲线（可选）
python scripts/plot_ablation.py /root/workspace/tmp/largemodel/outputs
```

### 📈 消融实验结果对比

| exp                       |   BLEU4 |   ROUGE1 |   ROUGEL |
|:--------------------------|--------:|---------:|---------:|
| ablate_no_label_smoothing |       0 |     0.07 |     0.04 |
| ablate_no_posenc          |       0 |     2.78 |     2.19 |
| ablate_single_head        |       0 |     5.58 |     4.55 |
| baseline                  |       0 |    10.06 |     6.99 |

| 模型配置 | BLEU-1 | BLEU-2 | ROUGE-1 (F1) | ROUGE-L (F1) |
|---------|:------:|:------:|:------------:|:------------:|
| **Baseline** | ~12 | ~0.02 | ~13 | ~11 |
| **w/o PosEnc** | ↓ | ↓ | ↓↓↓ | ↓↓↓ |
| **Single-Head** | ↓ | ↓ | ↓ | ↓ |
| **w/o Label Smoothing** | Loss ↓ but BLEU/ROUGE 崩溃 | 崩溃 | 崩溃 | 崩溃 |
⸻

## 📜 引用

主要参考的文章：

@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and others},
  journal={NeurIPS},
  year={2017}
}
