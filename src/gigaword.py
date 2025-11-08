import os, json, glob, csv
import torch
from tokenizer import build_vocab, encode, PAD

# 顶部：保留 import os, json, glob, csv 等
PAIR_PATTERNS = [
    # 常见配对
    (".article.txt", ".title.txt"),
    (".source.txt",  ".target.txt"),
    (".src.txt",     ".tgt.txt"),
    (".article",     ".title"),
    (".source",      ".target"),
    (".src",         ".tgt"),
    # 固定文件名（同目录）
    ("article.txt",  "title.txt"),
    ("source.txt",   "target.txt"),
    ("src.txt",      "tgt.txt"),
    # 你的 valid.*.filter.txt 命名
    (".article.filter.txt", ".title.filter.txt"),
]

def iter_pairs_from_dir(path: str):
    import io, re

    # 1) 先处理 JSONL/JSON/TSV（与之前一致）
    for f in glob.glob(os.path.join(path, '**', '*'), recursive=True):
        if os.path.isdir(f):
            continue
        low = f.lower()

        if low.endswith('.jsonl'):
            with open(f,'r',encoding='utf-8') as fp:
                for line in fp:
                    line=line.strip()
                    if not line: continue
                    obj = json.loads(line)
                    src = obj.get('src') or obj.get('source') or obj.get('article') or obj.get('text') or ''
                    tgt = obj.get('tgt') or obj.get('target') or obj.get('title')  or obj.get('summary') or ''
                    if src and tgt:
                        yield src, tgt
            continue

        if low.endswith('.json'):
            try:
                arr = json.load(open(f,'r',encoding='utf-8'))
                if isinstance(arr, list):
                    for obj in arr:
                        src = obj.get('src') or obj.get('source') or obj.get('article') or obj.get('text') or ''
                        tgt = obj.get('tgt') or obj.get('target') or obj.get('title')  or obj.get('summary') or ''
                        if src and tgt:
                            yield src, tgt
            except Exception:
                pass
            continue

        if low.endswith('.tsv'):
            with open(f,'r',encoding='utf-8') as fp:
                for row in csv.reader(fp, delimiter='\t'):
                    if len(row) >= 2:
                        yield row[0], row[1]
            continue

    # 2) 成对 txt：后缀/固定名匹配（含 *.article.filter.txt）
    files = [f for f in glob.glob(os.path.join(path, '**', '*'), recursive=True) if os.path.isfile(f)]
    files_set = set(files)
    used = set()

    def try_pair(a_suf, b_suf):
        for f in files:
            if f in used:
                continue
            lf = f.lower()
            if lf.endswith(a_suf):
                # 情况 A：前缀替换
                paired = f[:-len(a_suf)] + b_suf
                # 情况 B：固定文件名（同目录直接找 b_suf）
                if not os.path.exists(paired):
                    base_dir = os.path.dirname(f)
                    candidate = os.path.join(base_dir, b_suf)
                    if os.path.exists(candidate):
                        paired = candidate
                if os.path.exists(paired):
                    used.add(f); used.add(paired)
                    with open(f,'r',encoding='utf-8') as fs, open(paired,'r',encoding='utf-8') as ft:
                        for s,t in zip(fs,ft):
                            s=s.strip(); t=t.strip()
                            if s and t:
                                yield s, t

    for a,b in PAIR_PATTERNS:
        for s,t in try_pair(a,b):
            yield s,t

    # 3) DUC/Giga 风格：input.txt + task1_ref*.txt（选择最小编号的 ref 做 target）
    by_dir = {}
    for f in files:
        d = os.path.dirname(f)
        by_dir.setdefault(d, []).append(f)

    for d, fnames in by_dir.items():
        lower = {os.path.basename(x).lower(): x for x in fnames}
        if "input.txt" in lower:
            # 找到 task1_ref{0..9}.txt 中编号最小且存在的文件
            ref_path = None
            for i in range(10):
                name = f"task1_ref{i}.txt"
                if name in lower:
                    ref_path = lower[name]
                    break
            if ref_path is not None:
                inp = lower["input.txt"]
                with open(inp, 'r', encoding='utf-8') as fs, open(ref_path, 'r', encoding='utf-8') as ft:
                    for s, t in zip(fs, ft):
                        s=s.strip(); t=t.strip()
                        if s and t:
                            yield s, t

    # 4) 单文件 .txt：每行两列的备选格式（src \t tgt 或 src ||| tgt）
    for f in files:
        if f in used:
            continue
        if f.lower().endswith('.txt'):
            with open(f,'r',encoding='utf-8') as fp:
                for line in fp:
                    line=line.strip()
                    if not line:
                        continue
                    if '\t' in line:
                        parts=line.split('\t')
                        if len(parts)>=2:
                            yield parts[0], parts[1]
                            continue
                    if ' ||| ' in line:
                        parts=line.split(' ||| ')
                        if len(parts)>=2:
                            yield parts[0], parts[1]
                            continue

def build_or_load_vocab(train_dir: str, vocab_path: str, vocab_size: int,
                        rebuild: bool = False, min_ok_size: int = 100):
    """
    如果 vocab.json 存在且大小>=min_ok_size，则直接加载；
    否则（或 --rebuild_vocab 为 True）重建词表。
    """
    if (not rebuild) and os.path.exists(vocab_path):
        obj = json.load(open(vocab_path, 'r', encoding='utf-8'))
        itos = obj.get('itos', [])
        stoi = obj.get('stoi', {})
        if isinstance(itos, list) and len(itos) >= min_ok_size and isinstance(stoi, dict):
            return stoi, itos
        else:
            print(f"[Vocab] Existing vocab too small ({len(itos)}). Rebuilding...")

    texts = []
    cnt = 0
    for src, tgt in iter_pairs_from_dir(train_dir):
        texts.append(src); texts.append(tgt)
        cnt += 1
    if cnt == 0:
        raise RuntimeError(
            f"[Data] No pairs found in {train_dir}. "
            "Please check naming/format."
        )

    stoi, itos = build_vocab(texts, vocab_size=vocab_size)
    os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
    json.dump({'stoi': stoi, 'itos': itos}, open(vocab_path, 'w', encoding='utf-8'))
    return stoi, itos

def load_dataset(data_dir: str, stoi: dict, max_len: int, limit: int | None = None):
    X, Y_in, Y_out = [], [], []
    n = 0
    for src, tgt in iter_pairs_from_dir(data_dir):
        x = encode(src, stoi, max_len)
        y = encode(tgt, stoi, max_len)

        # 兼容 encode 返回 list[int] 或 Tensor 的两种情况
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.long)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.long)

        # y_in = y 左移一位；末尾补 PAD（而不是 + [0]）
        y_in = torch.cat([y[:-1], torch.tensor([PAD], dtype=torch.long)], dim=0)

        X.append(x)
        Y_in.append(y_in)
        Y_out.append(y)

        n += 1
        if limit is not None and n >= limit:
            break

    if len(X) == 0:
        # 返回空张量，避免 DataLoader 继续报错
        empty = torch.empty(0, max_len, dtype=torch.long)
        return empty, empty, empty

    return torch.stack(X, dim=0), torch.stack(Y_in, dim=0), torch.stack(Y_out, dim=0)

# 调试：列出扫描统计，便于你确认到底读到了什么
def debug_scan(data_dir:str, max_show:int=5):
    import collections
    all_files = [f for f in glob.glob(os.path.join(data_dir, '**', '*'), recursive=True) if os.path.isfile(f)]
    by_ext = collections.Counter([os.path.splitext(f)[1].lower() for f in all_files])
    print(f"[Data] scan dir={data_dir}")
    print(f"[Data] total files={len(all_files)} by ext={dict(by_ext)}")
    # 抽样输出几条
    sample_pairs = []
    for i,(s,t) in enumerate(iter_pairs_from_dir(data_dir)):
        if i < max_show:
            sample_pairs.append((s[:80], t[:80]))
        else:
            break
    print(f"[Data] sample pairs found={len(sample_pairs)} (show up to {max_show}):")
    for i,(s,t) in enumerate(sample_pairs):
        print(f"  - #{i+1} SRC: {s}")
        print(f"        TGT: {t}")