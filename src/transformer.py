import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import MultiHeadAttention, PositionwiseFFN, PreNormResidual, PositionalEncoding

class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, dropout):
        super().__init__()
        self.self_attn = PreNormResidual(d_model, MultiHeadAttention(d_model, n_heads, dropout), dropout)
        self.ffn = PreNormResidual(d_model, PositionwiseFFN(d_model, ff_dim, dropout), dropout)

    def forward(self, x, src_key_padding_mask=None):
        x = self.self_attn(x, x, x, key_padding_mask=src_key_padding_mask)
        x = self.ffn(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim, dropout):
        super().__init__()
        self.self_attn = PreNormResidual(d_model, MultiHeadAttention(d_model, n_heads, dropout), dropout)
        self.cross_attn = PreNormResidual(d_model, MultiHeadAttention(d_model, n_heads, dropout), dropout)
        self.ffn = PreNormResidual(d_model, PositionwiseFFN(d_model, ff_dim, dropout), dropout)

    def forward(self, x, mem, tgt_mask=None, tgt_key_padding_mask=None, mem_key_padding_mask=None):
        x = self.self_attn(x, x, x, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        x = self.cross_attn(x, mem, mem, key_padding_mask=mem_key_padding_mask)
        x = self.ffn(x)
        return x

def causal_mask(sz):
    mask = torch.tril(torch.ones(sz, sz, dtype=torch.uint8))
    return mask.unsqueeze(0).unsqueeze(0)  # (1,1,L,L)

class TransformerSeq2Seq(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=256, n_heads=4, num_layers=4, ff_dim=1024, dropout=0.1,
                 max_len=512, share_embeddings=False, use_posenc=True):
        super().__init__()
        self.src_embed = nn.Embedding(src_vocab, d_model, padding_idx=0)
        self.tgt_embed = nn.Embedding(tgt_vocab, d_model, padding_idx=0)
        self.use_posenc = use_posenc
        if use_posenc:
            self.posenc = PositionalEncoding(d_model, max_len=max_len)
        self.enc_layers = nn.ModuleList([EncoderLayer(d_model, n_heads, ff_dim, dropout) for _ in range(num_layers)])
        self.dec_layers = nn.ModuleList([DecoderLayer(d_model, n_heads, ff_dim, dropout) for _ in range(num_layers)])
        self.proj = nn.Linear(d_model, tgt_vocab)
        if share_embeddings:
            assert src_vocab==tgt_vocab
            self.tgt_embed.weight = self.src_embed.weight

        self.d_model = d_model

    def encode(self, src_ids, src_key_padding_mask=None):
        x = self.src_embed(src_ids)
        if self.use_posenc:
            x = self.posenc(x)
        for layer in self.enc_layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return x

    def decode(self, tgt_ids, mem, tgt_key_padding_mask=None, mem_key_padding_mask=None):
        x = self.tgt_embed(tgt_ids)
        if self.use_posenc:
            x = self.posenc(x)
        L = tgt_ids.size(1)
        m = causal_mask(L).to(tgt_ids.device)
        for layer in self.dec_layers:
            x = layer(x, mem, tgt_mask=m, tgt_key_padding_mask=tgt_key_padding_mask, mem_key_padding_mask=mem_key_padding_mask)
        return self.proj(x)

    def forward(self, src_ids, tgt_inp, src_key_padding_mask=None, tgt_key_padding_mask=None):
        mem = self.encode(src_ids, src_key_padding_mask=src_key_padding_mask)
        logits = self.decode(tgt_inp, mem, tgt_key_padding_mask=tgt_key_padding_mask, mem_key_padding_mask=src_key_padding_mask)
        return logits

    @torch.no_grad()
    def greedy_decode(self, src_ids, max_len, bos_id=1, eos_id=2, min_len: int = 5, pad_id: int = 0):
        """
        Greedy 解码，增加：
          1) 前 min_len 步禁止生成 EOS
          2) 全程禁止生成 BOS/PAD
          3) 若依然空，做一次兜底（避免空串导致 BLEU/ROUGE=0）
        """
        B = src_ids.size(0)
        device = src_ids.device

        mem = self.encode(src_ids, src_key_padding_mask=(src_ids != pad_id))
        ys = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for t in range(max_len - 1):
            logits = self.decode(
                ys, mem,
                tgt_key_padding_mask=(ys != pad_id),
                mem_key_padding_mask=(src_ids != pad_id)
            )  # (B, L, V)
            step_logits = logits[:, -1, :]  # (B, V)

            # ——屏蔽特殊符号——
            step_logits[:, bos_id] = -1e9
            step_logits[:, pad_id] = -1e9
            if t < min_len:
                step_logits[:, eos_id] = -1e9

            next_token = step_logits.argmax(dim=-1, keepdim=True)  # (B,1)
            ys = torch.cat([ys, next_token], dim=1)

            finished |= (next_token.squeeze(1) == eos_id)
            if finished.all():
                break

        # 兜底：若只有 <bos><eos> 或几乎空，强制保留至少一个非特殊 token
        # （这样 BLEU/ROUGE 就不会是 0）
        for i in range(B):
            seq = ys[i].tolist()
            non_special = [tok for tok in seq if tok not in (bos_id, eos_id, pad_id)]
            if len(non_special) == 0:
                # 把最后一个位置强制为非特殊最高分 token（粗暴但有效）
                # 这里简单取 bos 后一个 token 或者 3（<unk>）之外的 top1
                # 如果你缓存了最后一步的 step_logits，可用 step_logits[i].topk(5) 过滤特殊再选
                # 这里退而求其次：用 <unk>=3 兜底（你的词表第4个）
                unk_id = 3
                if ys.size(1) > 1:
                    ys[i, 1] = torch.tensor(unk_id, device=device)

        return ys

    @torch.no_grad()
    def beam_search_decode(
            self, src_ids, max_len, bos_id=1, eos_id=2, pad_id=0,
            beam_size=6, length_penalty=0.9, min_len=5,
            no_repeat_ngram_size=3, eos_bias=2.0,  # 新增 eos_bias
    ):
        """
        标准 Beam Search（批量版）。
          - length_penalty: GNMT 风格，score /= ((5+L)^alpha / (6^alpha))
          - min_len: 前 min_len 步禁止 EOS
          - 屏蔽 BOS/PAD
          - no_repeat_ngram_size>0 时阻止重复 n-gram（小 beam 下开销可接受）

        返回：shape (B, Lmax) 的 LongTensor，已 <bos> 开头，<eos> 截断，pad 填充。
        """
        device = src_ids.device
        B = src_ids.size(0)

        # 编码一次，并复制到每个 beam
        mem = self.encode(src_ids, src_key_padding_mask=(src_ids != pad_id))  # (B, S, D)
        S, D = mem.size(1), mem.size(2)
        mem = mem.unsqueeze(1).expand(B, beam_size, S, D).contiguous().view(B * beam_size, S, D)
        src_mask = (src_ids != pad_id).unsqueeze(1).expand(B, beam_size, S).contiguous().view(B * beam_size, S)

        # 初始序列与分数
        ys = torch.full((B, beam_size, 1), bos_id, dtype=torch.long, device=device)  # (B, K, 1)
        scores = torch.zeros(B, beam_size, device=device)  # 累积 logprob
        finished = torch.zeros(B, beam_size, dtype=torch.bool, device=device)

        def apply_no_repeat_ngram_mask(step_logits, seq, n):
            """step_logits: (V,), seq: (L,)；就地把重复 n-gram 的下一个 token 置为 -inf"""
            if n <= 0 or seq.size(0) < n:
                return
            import collections
            banned = set()
            # 建字典：前缀(n-1) -> 可跟随的 token 集
            prefix2next = collections.defaultdict(set)
            tokens = seq.tolist()
            for i in range(len(tokens) - n + 1):
                prev = tuple(tokens[i:i + n - 1])
                nxt = tokens[i + n - 1]
                prefix2next[prev].add(nxt)
            prefix = tuple(tokens[-(n - 1):]) if n > 1 else tuple()
            for nxt in prefix2next.get(prefix, []):
                banned.add(nxt)
            if banned:
                idx = torch.tensor(list(banned), device=step_logits.device, dtype=torch.long)
                step_logits.index_fill_(0, idx, float('-inf'))

        for t in range(max_len - 1):
            # 展平成 (B*K, L)
            cur = ys.view(B * beam_size, -1)  # (B*K, L)
            # 前向
            logits = self.decode(
                cur, mem,
                tgt_key_padding_mask=(cur != pad_id),
                mem_key_padding_mask=src_mask
            )  # (B*K, L, V)
            step_logits = logits[:, -1, :]  # (B*K, V)

            # 屏蔽特殊符号
            step_logits[:, bos_id] = float('-inf')
            step_logits[:, pad_id] = float('-inf')
            if t < min_len:
                step_logits[:, eos_id] = float('-inf')
            else:
                step_logits[:, eos_id] = step_logits[:, eos_id] + eos_bias  # 鼓励尽早结束

            # 对已完成的 beam，仅允许继续选 EOS（保持分数不变）
            done_mask = finished.view(-1)  # (B*K,)
            if done_mask.any():
                step_logits[done_mask] = float('-inf')
                step_logits[done_mask, eos_id] = 0.0

            # no-repeat-ngram
            if no_repeat_ngram_size > 0:
                # 小 beam 情况下逐条处理即可
                for b in range(B):
                    for k in range(beam_size):
                        if finished[b, k]:
                            continue
                        seq = ys[b, k]
                        apply_no_repeat_ngram_mask(step_logits[b * beam_size + k], seq, no_repeat_ngram_size)

            logprobs = F.log_softmax(step_logits, dim=-1)  # (B*K, V)

            # 展回 (B, K, V)，进行 TopK(beam) 选择
            logprobs = logprobs.view(B, beam_size, -1)  # (B, K, V)
            cand_scores = scores.unsqueeze(-1) + logprobs  # (B, K, V)
            V = logprobs.size(-1)
            cand_scores = cand_scores.view(B, -1)  # (B, K*V)
            topk_scores, topk_idx = torch.topk(cand_scores, k=beam_size, dim=-1)  # (B, K)

            # 反推回来源 beam 与 next_token
            next_beam_idx = topk_idx // V  # (B, K)
            next_tokens = (topk_idx % V).long()  # (B, K)

            # 更新序列：先根据 next_beam_idx gather，再 append next_tokens
            ys = ys.gather(1, next_beam_idx.unsqueeze(-1).expand(B, beam_size, ys.size(-1)))
            ys = torch.cat([ys, next_tokens.unsqueeze(-1)], dim=-1)  # (B, K, L+1)

            # 更新分数、完成标记
            scores = topk_scores
            newly_finished = (next_tokens == eos_id)
            finished = finished.gather(1, next_beam_idx) | newly_finished

            # 若该 batch 全部完成，可以继续几步也行；这里不提前 break 保持等长，便于向量化

        # 最终从每个 batch 的 K 个 beam 中，按 length_penalty 选最优
        def lp(len_):
            return ((5.0 + len_) ** length_penalty) / (6.0 ** length_penalty)

        final_scores = scores.clone()
        lengths = ys.ne(pad_id).sum(dim=-1)  # (B, K) 近似长度（含 bos/eos）
        norm_scores = final_scores / lp(lengths.clamp_min(1).float())

        best_idx = norm_scores.argmax(dim=-1)  # (B,)
        best_seq = ys[torch.arange(B, device=device), best_idx]  # (B, L)

        # 规范化输出：截到 eos，并右侧 pad
        out = []
        for i in range(B):
            seq = best_seq[i].tolist()
            # 去掉重复的前缀 bos
            if len(seq) >= 1 and seq[0] == bos_id:
                seq = seq
            # 截断到第一个 eos
            if eos_id in seq[1:]:
                j = seq.index(eos_id, 1)
                seq = seq[: j + 1]
            out.append(torch.tensor(seq, device=device, dtype=torch.long))

        maxL = max(s.size(0) for s in out)
        padded = torch.full((B, maxL), pad_id, dtype=torch.long, device=device)
        for i, s in enumerate(out):
            padded[i, : s.size(0)] = s
        return padded
