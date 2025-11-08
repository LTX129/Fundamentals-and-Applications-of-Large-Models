from collections import Counter
from typing import List, Dict, Tuple
import math

import re

import re

_WS = re.compile(r"\s+")
# Gigaword/DUC 常见括号与引号 token
_PUN_MAP = {
    "-lrb-": "(", "-rrb-": ")",
    "-lsb-": "[", "-rsb-": "]",
    "-lcb-": "{", "-rcb-": "}",
    "``": '"', "''": '"',
}

# 把所有数字规整为占位符 N（或你喜欢的 0），避免 12/12.3/#.## 之类让 n-gram 匹配失败
_NUM = re.compile(r"\d+(\.\d+)?")
def normalize_for_eval(s: str) -> str:
    s = s.lower().strip()
    for k, v in _PUN_MAP.items():
        s = s.replace(k, v)
    s = _NUM.sub("n", s)         # ← 关键：数字归一化
    s = _WS.sub(" ", s)
    return s


def _ngram_counts(tokens:List[str], n:int)->Counter:
    return Counter([tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)])

def corpus_bleu(references:List[List[str]], hypotheses:List[List[str]], max_n:int=4, smooth_eps:float=1e-9)->float:
    # references: list of list-of-tokens (single ref per sample assumed for simplicity)
    weights = [1.0/max_n]*max_n
    p_ns = []
    for n in range(1, max_n+1):
        num=0; den=0
        for ref, hyp in zip(references, hypotheses):
            ref_counts = _ngram_counts(ref, n)
            hyp_counts = _ngram_counts(hyp, n)
            overlap = {g: min(c, ref_counts.get(g,0)) for g,c in hyp_counts.items()}
            num += sum(overlap.values())
            den += max(1, sum(hyp_counts.values()))
        p = (num + smooth_eps) / (den + smooth_eps)
        p_ns.append(p)
    # brevity penalty
    ref_len = sum(len(r) for r in references)
    hyp_len = sum(len(h) for h in hypotheses)
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / max(1,hyp_len))
    score = bp * math.exp(sum(w*math.log(p) for w,p in zip(weights,p_ns)))

    return score * 100.0

def corpus_bleu_n(references, hypotheses, max_n=4, smooth_eps=1e-9):
    # 返回各阶 BLEU-n 以及累积 BLEU-4
    import math
    def _ngram_counts(tokens, n):
        from collections import Counter
        return Counter([tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)])

    bleu_ns = []
    for n in range(1, max_n+1):
        num = den = 0
        for ref, hyp in zip(references, hypotheses):
            rc = _ngram_counts(ref, n)
            hc = _ngram_counts(hyp, n)
            overlap = {g: min(c, rc.get(g, 0)) for g, c in hc.items()}
            num += sum(overlap.values())
            den += max(1, sum(hc.values()))
        p = (num + smooth_eps) / (den + smooth_eps)
        bleu_ns.append(p)

    # BP
    ref_len = sum(len(r) for r in references)
    hyp_len = sum(len(h) for h in hypotheses)
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / max(1, hyp_len))

    cum_bleu4 = bp * math.exp(sum((1/4)*math.log(p) for p in bleu_ns[:4]))
    # 返回小数（非百分数）；是否乘以100由外层决定
    return {
        "bleu1": bleu_ns[0],
        "bleu2": bleu_ns[1] if max_n>=2 else None,
        "bleu3": bleu_ns[2] if max_n>=3 else None,
        "bleu4": bleu_ns[3] if max_n>=4 else None,
        "bleu_cum4": cum_bleu4
    }

def _lcs(x:List[str], y:List[str])->int:
    m,n=len(x),len(y)
    dp=[[0]*(n+1) for _ in range(m+1)]
    for i in range(m):
        for j in range(n):
            if x[i]==y[j]:
                dp[i+1][j+1]=dp[i][j]+1
            else:
                dp[i+1][j+1]=max(dp[i][j+1], dp[i+1][j])
    return dp[m][n]

def rouge_scores(references:List[List[str]], hypotheses:List[List[str]]):
    def rouge_n(n):
        num=0; den=0
        num_p=0; den_p=0
        for ref,hyp in zip(references, hypotheses):
            ref_c=_ngram_counts(ref,n); hyp_c=_ngram_counts(hyp,n)
            overlap = sum((ref_c & hyp_c).values())
            num += overlap
            den += max(1,sum(ref_c.values()))
            num_p += overlap
            den_p += max(1,sum(hyp_c.values()))
        rec = num/den
        prec = num_p/den_p
        f1 = 0 if (rec+prec)==0 else 2*rec*prec/(rec+prec)
        return rec*100.0, f1*100.0
    r1, f1_1 = rouge_n(1)
    r2, f1_2 = rouge_n(2)
    # ROUGE-L via LCS (recall & F1)
    recs=[]; precs=[]
    for ref,hyp in zip(references, hypotheses):
        l=_lcs(ref,hyp)
        recs.append(l/max(1,len(ref)))
        precs.append(l/max(1,len(hyp)))
    rec=sum(recs)/len(recs)
    prec=sum(precs)/len(precs)
    f1=0 if (rec+prec)==0 else 2*rec*prec/(rec+prec)
    return {
        "rouge1_recall": r1, "rouge1_f1": f1_1,
        "rouge2_recall": r2, "rouge2_f1": f1_2,
        "rougeL_recall": rec*100.0, "rougeL_f1": f1*100.0
    }
def corpus_bleu_multi_ref(list_of_references, hypotheses, max_n=4, smooth_eps=1e-9):
    """
    list_of_references: List[N] -> List[K refs] -> List[tokens]
    hypotheses:         List[N] -> List[tokens]
    返回 cum BLEU-4（0~1）
    """
    import math
    def _ngram_counts(tokens, n):
        from collections import Counter
        return Counter([tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)])

    p_ns = []
    for n in range(1, max_n+1):
        num = den = 0
        for refs, hyp in zip(list_of_references, hypotheses):
            hc = _ngram_counts(hyp, n)
            # 多参考取 max 匹配
            max_overlap = {}
            for rcand in refs:
                rc = _ngram_counts(rcand, n)
                for g, c in hc.items():
                    max_overlap[g] = max(max_overlap.get(g, 0), min(c, rc.get(g, 0)))
            num += sum(max_overlap.values())
            den += max(1, sum(hc.values()))
        p = (num + smooth_eps) / (den + smooth_eps)
        p_ns.append(p)

    # BP：按每条参考长度中离 hypothes 最近的那个（近似）
    ref_len = 0
    hyp_len = 0
    for refs, hyp in zip(list_of_references, hypotheses):
        hyp_len += len(hyp)
        ref_lens = [len(r) for r in refs]
        ref_len += sorted(ref_lens, key=lambda x: abs(x-len(hyp)))[0]
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / max(1, hyp_len))
    bleu = bp * math.exp(sum((1/4)*math.log(p) for p in p_ns))
    return bleu

def rouge_scores_multi_ref(list_of_references, hypotheses):
    """
    返回 dict: rouge1_f1 / rougeL_f1（百分比）
    多参考下取最大值（常见做法）。
    """
    from collections import defaultdict
    def rouge_f1(ref, hyp):
        ref_set = ref
        hyp_set = hyp
        import difflib
        # ROUGE-1 近似：unigram overlap
        ref_unis = set(ref_set)
        hyp_unis = set(hyp_set)
        overlap = len(ref_unis & hyp_unis)
        r1_p = overlap / max(1, len(hyp_unis))
        r1_r = overlap / max(1, len(ref_unis))
        r1_f = 0.0 if (r1_p+r1_r)==0 else 2*r1_p*r1_r/(r1_p+r1_r)
        # ROUGE-L 近似：LCS
        sm = difflib.SequenceMatcher(None, ref, hyp)
        lcs = int(round(sum(tr.size for tr in sm.get_matching_blocks())))
        rl_p = lcs / max(1, len(hyp))
        rl_r = lcs / max(1, len(ref))
        rl_f = 0.0 if (rl_p+rl_r)==0 else 2*rl_p*rl_r/(rl_p+rl_r)
        return r1_f*100.0, rl_f*100.0

    r1_best, rl_best = [], []
    for refs, hyp in zip(list_of_references, hypotheses):
        r1s, rls = [], []
        for r in refs:
            r1, rl = rouge_f1(r, hyp)
            r1s.append(r1); rls.append(rl)
        r1_best.append(max(r1s)); rl_best.append(max(rls))
    return {
        "rouge1_f1": sum(r1_best)/len(r1_best),
        "rougeL_f1": sum(rl_best)/len(rl_best),
    }

def corpus_bleu_report_multi_ref(list_of_references, hypotheses, max_n=4, smooth_eps=1e-9):
    """
    多参考 BLEU 报表：返回 {bleu1, bleu2, bleu3, bleu4, bleu_cum4}，数值范围 0~1（非百分比）。
    list_of_references: List[N] -> List[K] -> List[tokens]
    hypotheses:         List[N] -> List[tokens]
    """
    import math
    from collections import Counter

    def ngram_counts(tokens, n):
        return Counter([tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)])

    p_ns = []
    for n in range(1, max_n+1):
        overlap_sum, hyp_sum = 0, 0
        for refs, hyp in zip(list_of_references, hypotheses):
            hc = ngram_counts(hyp, n)
            # 多参考：对每个 n-gram 取最大匹配
            max_overlap = {}
            for r in refs:
                rc = ngram_counts(r, n)
                for g, c in hc.items():
                    mc = min(c, rc.get(g, 0))
                    if mc > max_overlap.get(g, 0):
                        max_overlap[g] = mc
            overlap_sum += sum(max_overlap.values())
            hyp_sum     += max(1, sum(hc.values()))
        p = (overlap_sum + smooth_eps) / (hyp_sum + smooth_eps)
        p_ns.append(p)

    # BP：为每条样本选长度最接近 hyp 的参考
    ref_len, hyp_len = 0, 0
    for refs, hyp in zip(list_of_references, hypotheses):
        hyp_len += len(hyp)
        ref_len += min((abs(len(r) - len(hyp)), len(r)) for r in refs)[1]
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / max(1, hyp_len))

    cum4 = bp * math.exp(sum((1/max_n) * math.log(p) for p in p_ns[:max_n]))
    out = {f"bleu{i}": p_ns[i-1] for i in range(1, max_n+1)}
    out["bleu_cum4"] = cum4
    return out