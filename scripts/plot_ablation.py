#!/usr/bin/env python3
import os, sys, pandas as pd
import matplotlib.pyplot as plt

OUT_ROOT = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("OUT_ROOT", "../results/outputs")
csv_path = os.path.join(OUT_ROOT, "ablation_summary.csv")
if not os.path.exists(csv_path):
    print("Run collect_results.py first.")
    sys.exit(0)

df = pd.read_csv(csv_path)

plt.figure(figsize=(7,5))
for exp, g in df.groupby("exp"):
    plt.plot(g["epoch"], g["BLEU4"], marker="o", label=exp)
plt.xlabel("Epoch"); plt.ylabel("BLEU-4"); plt.title("BLEU-4 vs Epoch")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT_ROOT, "bleu4_vs_epoch.png"))

plt.figure(figsize=(7,5))
for exp, g in df.groupby("exp"):
    plt.plot(g["epoch"], g["ROUGE1"], marker="o", label=exp)
plt.xlabel("Epoch"); plt.ylabel("ROUGE-1 (F1)"); plt.title("ROUGE-1 vs Epoch")
plt.legend(); plt.tight_layout()
plt.savefig(os.path.join(OUT_ROOT, "rouge1_vs_epoch.png"))

print("Saved plots to", OUT_ROOT)