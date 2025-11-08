import re
from collections import Counter
from typing import List, Tuple
import torch
from torch import Tensor

PAD=0; BOS=1; EOS=2; UNK=3
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]

def basic_tokenize(text:str)->List[str]:
    text = text.strip().lower()
    # split on whitespace and punctuation
    tokens = re.findall(r"[\w]+|[^\w\s]", text, flags=re.UNICODE)
    return tokens

def build_vocab(texts, vocab_size:int=32000):
    counter=Counter()
    for t in texts:
        counter.update(basic_tokenize(t))
    most = [w for w,_ in counter.most_common(vocab_size - len(SPECIAL_TOKENS))]
    itos = SPECIAL_TOKENS + most
    stoi = {w:i for i,w in enumerate(itos)}
    return stoi, itos

def encode(text:str, stoi, max_len:int)-> Tensor:
    toks = basic_tokenize(text)
    ids = [BOS] + [stoi.get(t, UNK) for t in toks][:max_len-2] + [EOS]
    if len(ids) < max_len:
        ids = ids + [PAD]*(max_len-len(ids))

    ids = ids[:max_len] + [PAD] * max(0, max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)

def decode(ids:List[int], itos)->str:
    toks=[]
    for i in ids:
        if i==PAD: continue
        if i==BOS or i==EOS: continue
        toks.append(itos[i] if 0<=i<len(itos) else "<unk>")
    # simple detok: join with space then compact punctuation
    s = " ".join(toks)
    s = re.sub(r"\s+([.,!?;:])", r"\1", s)
    return s
