

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
set_seed(3407)
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
