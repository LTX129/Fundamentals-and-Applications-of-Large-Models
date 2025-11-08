#!/usr/bin/env python3
import re, sys, os, glob, pandas as pd

OUT_ROOT = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("OUT_ROOT", "../results/outputs")

pat = re.compile(
    r"Eval BLEU-4(?:\(mref\))?:\s*([0-9.]+)\s*\|\s*ROUGE-1\(F1(?:,mref)?\):\s*([0-9.]+)\s*\|\s*ROUGE-L\(F1(?:,mref)?\):\s*([0-9.]+)"
)
ep_pat = re.compile(r"^epoch\s+(\d+):")

rows = []
for exp_dir in sorted(glob.glob(os.path.join(OUT_ROOT, "*"))):
    log = os.path.join(exp_dir, "train.log")
    if not os.path.exists(log):
        continue
    exp = os.path.basename(exp_dir)
    epoch = None
    with open(log, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            m_ep = ep_pat.search(ln)
            if m_ep:
                epoch = int(m_ep.group(1))
            m = pat.search(ln)
            if m and epoch is not None:
                rows.append({
                    "exp": exp,
                    "epoch": epoch,
                    "BLEU4": float(m.group(1)),
                    "ROUGE1": float(m.group(2)),
                    "ROUGEL": float(m.group(3)),
                })

df = pd.DataFrame(rows).sort_values(["exp","epoch"])
if df.empty:
    print("No results found under", OUT_ROOT)
    sys.exit(0)

csv_path = os.path.join(OUT_ROOT, "ablation_summary.csv")
md_path  = os.path.join(OUT_ROOT, "ablation_summary.md")
df.to_csv(csv_path, index=False)

last = df.sort_values("epoch").groupby("exp").tail(1)[["exp","BLEU4","ROUGE1","ROUGEL"]]
last = last.sort_values("BLEU4", ascending=False)

with open(md_path, "w", encoding="utf-8") as f:
    f.write("# Ablation Summary (final epoch)\n\n")
    f.write(last.to_markdown(index=False))
    f.write("\n")

print("Saved:", csv_path)
print("Saved:", md_path)