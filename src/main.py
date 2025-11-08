import os, argparse, json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from metering import SmoothedValue, Checkpointer
from schedule import InverseSqrtWithWarmup
from gigaword import build_or_load_vocab, load_dataset
from transformer import TransformerSeq2Seq
from bleu_rouge import corpus_bleu, rouge_scores, corpus_bleu_n, normalize_for_eval, corpus_bleu_report_multi_ref
from gigaword import debug_scan  # 新增引入

import os, glob, re
from tokenizer import encode, decode, PAD, BOS, EOS
from bleu_rouge import corpus_bleu_multi_ref, rouge_scores_multi_ref


def plot_curves(history, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # loss
    plt.figure()
    plt.plot(range(len(history['train_loss'])), history['train_loss'])
    plt.xlabel('step'); plt.ylabel('loss'); plt.title('Train Loss')
    plt.savefig(out_path.replace('.png','_loss.png')); plt.close()
    # lr
    plt.figure()
    plt.plot(range(len(history['lrs'])), history['lrs'])
    plt.xlabel('step'); plt.ylabel('lr'); plt.title('Learning Rate')
    plt.savefig(out_path.replace('.png','_lr.png')); plt.close()

def tokenize_to_list(ids, itos):
    return [w for w in decode(ids, itos).split()]

def run_eval(model, dl, itos, device, max_len):
    model.eval()
    refs, hyps = [], []
    with torch.no_grad():
        for xb, y_in, y_out in tqdm(dl, desc='eval', leave=False):
            xb = xb.to(device)
            if args.decode == 'beam':
                gens = model.beam_search_decode(
                    xb, max_len=max_len, bos_id=BOS, eos_id=EOS, pad_id=PAD,
                    beam_size=args.beam_size, length_penalty=args.length_penalty,
                    min_len=args.min_gen_len, no_repeat_ngram_size=args.no_repeat_ngram_size,eos_bias=args.eos_bias,
                )
            else:
                gens = model.greedy_decode(
                    xb, max_len=max_len, bos_id=BOS, eos_id=EOS,
                    min_len=args.min_gen_len, pad_id=PAD
                )
            for i in range(xb.size(0)):
                ref = normalize_for_eval(decode(y_out[i].cpu().tolist(), itos)).split()
                hyp = normalize_for_eval(decode(gens[i].cpu().tolist(), itos)).split()
                refs.append(ref)
                hyps.append(hyp)

    avg_len = sum(len(h) for h in hyps) / max(1, len(hyps))
    empty_cnt = sum(1 for h in hyps if len(h) == 0)
    print(f"[eval] avg_hyp_len={avg_len:.2f}, empty_hyp={empty_cnt}/{len(hyps)}")

    bleu_dict = corpus_bleu_n(refs, hyps, max_n=4, smooth_eps=1e-9)
    rouge = rouge_scores(refs, hyps)  # 这是百分比（0-100）

    # 打印更多小数位，避免显示为 0.00
    print(
      f"BLEU-1:{bleu_dict['bleu1']*100:.3f} "
      f"BLEU-2:{bleu_dict['bleu2']*100:.3f} "
      f"BLEU-3:{bleu_dict['bleu3']*100:.3f} "
      f"BLEU-4:{bleu_dict['bleu4']*100:.3f} "
      f"Cum4:{bleu_dict['bleu_cum4']*100:.3f}"
    )

    # 你之前历史里用的是 BLEU-4（百分比）
    return bleu_dict['bleu_cum4']*100.0, rouge

def run_eval_ducref(model, valid_dir, stoi, itos, device, args):
    # --- 统一归一化，避免 BLEU 因大小写/括号记号不同而被压成 0 ---
    WS = re.compile(r"\s+")
    PUN = {
        "-lrb-": "(", "-rrb-": ")",
        "-lsb-": "[", "-rsb-": "]",
        "-lcb-": "{", "-rcb-": "}",
        "``": '"', "''": '"',
    }
    def normalize_for_eval(s: str) -> str:
        s = s.lower().strip()
        for k, v in PUN.items():
            s = s.replace(k, v)
        s = WS.sub(" ", s)
        return s

    # --- 文件就绪性检查 ---
    inp = os.path.join(valid_dir, "input.txt")
    refs = sorted(glob.glob(os.path.join(valid_dir, "task1_ref*.txt")))
    assert os.path.exists(inp) and refs, "[eval] DUC2004 multi-ref files not found."

    # --- 读取并裁齐 ---
    with open(inp, 'r', encoding='utf-8') as f:
        src_lines = [ln.strip() for ln in f if ln.strip()]
    ref_sets = []
    for r in refs:
        with open(r, 'r', encoding='utf-8') as f:
            ref_sets.append([ln.strip() for ln in f if ln.strip()])
    N = min(len(src_lines), *[len(r) for r in ref_sets])
    src_lines = src_lines[:N]
    ref_sets = [r[:N] for r in ref_sets]

    # --- 批量解码 ---
    B = args.batch_size
    hyps_tok = []
    for i in range(0, N, B):
        batch_src = src_lines[i:i+B]
        ids_list = [encode(s, stoi, args.max_len) for s in batch_src]
        if not isinstance(ids_list[0], torch.Tensor):
            ids_list = [torch.tensor(x, dtype=torch.long) for x in ids_list]
        X = torch.stack(ids_list, dim=0).to(device)

        if getattr(args, "decode", "beam") == 'beam':
            gens = model.beam_search_decode(
                X, max_len=args.max_len, bos_id=BOS, eos_id=EOS, pad_id=PAD,
                beam_size=getattr(args, "beam_size", 4),
                length_penalty=getattr(args, "length_penalty", 0.6),
                min_len=getattr(args, "min_gen_len", 5),
                no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
                eos_bias=args.eos_bias,
            )
        else:
            gens = model.greedy_decode(
                X, max_len=args.max_len, bos_id=BOS, eos_id=EOS, pad_id=PAD,
                min_len=getattr(args, "min_gen_len", 5)
            )

        # decode -> 归一化 -> 分词
        for g in gens:
            hyp_str = decode(g.tolist(), itos)
            hyp_str = normalize_for_eval(hyp_str)
            hyp_tok = hyp_str.split()
            hyps_tok.append(hyp_tok)

    # --- 多参考：对每条样本收集所有参考的 token 列表 ---
    refs_tok = []
    for j in range(N):
        refs_tok.append([normalize_for_eval(ref_sets[k][j]).split() for k in range(len(refs))])

    # --- 评测：多参考 BLEU/ROUGE ---
    bleu = corpus_bleu_multi_ref(refs_tok, hyps_tok) * 100.0  # 百分比
    rouge = rouge_scores_multi_ref(refs_tok, hyps_tok)        # 已是百分比

    avg_len = sum(len(h) for h in hyps_tok) / max(1, len(hyps_tok))
    empty_cnt = sum(1 for h in hyps_tok if len(h) == 0)
    print(f"[eval] avg_hyp_len={avg_len:.2f}, empty_hyp={empty_cnt}/{len(hyps_tok)}")
    # 多参考 BLEU 报表（分阶 + 累积）
    bleu_report = corpus_bleu_report_multi_ref(refs_tok, hyps_tok, max_n=4, smooth_eps=1e-9)
    # 多参考 ROUGE（你已有）
    rouge = rouge_scores_multi_ref(refs_tok, hyps_tok)

    print(
        "BLEU-1:{:.3f} BLEU-2:{:.3f} BLEU-3:{:.3f} BLEU-4:{:.3f} Cum4:{:.3f}".format(
            bleu_report["bleu1"] * 100.0,
            bleu_report["bleu2"] * 100.0,
            bleu_report["bleu3"] * 100.0,
            bleu_report["bleu4"] * 100.0,
            bleu_report["bleu_cum4"] * 100.0,
        )
    )
    print(
        "Eval BLEU-4(mref): {:.2f} | ROUGE-1(F1,mref): {:.2f} | ROUGE-L(F1,mref): {:.2f}".format(
            bleu_report["bleu_cum4"] * 100.0, rouge["rouge1_f1"], rouge["rougeL_f1"]
        )
    )
    return bleu, rouge

def train(args):
    # ---- 路径规范化（你已有的工具）----
    args.data_dir  = str(resolve_under_root(args.data_dir))
    args.valid_dir = str(resolve_under_root(args.valid_dir)) if args.valid_dir else None
    args.work_dir  = str(resolve_under_root(args.work_dir))
    args.ckpt_dir  = str(resolve_under_root(args.ckpt_dir))
    args.ckpt_path = str(resolve_under_root(args.ckpt_path))

    # ---- 设备/全局开关 ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.work_dir, exist_ok=True)
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True

    # ---- 词表 ----
    vocab_path = os.path.join(args.work_dir, 'vocab.json')
    stoi, itos = build_or_load_vocab(
        args.data_dir, vocab_path, args.vocab_size,
        rebuild=args.rebuild_vocab, min_ok_size=args.min_vocab_size
    )
    print(f"Vocab size: {len(itos)}")

    # ---- dry run: 只探测数据 ----
    if args.dry_run:
        debug_scan(args.data_dir)
        if args.valid_dir:
            debug_scan(args.valid_dir)
        return

    # ---- 数据加载 ----
    X, Y_in, Y_out = load_dataset(args.data_dir, stoi, args.max_len, limit=args.train_limit)
    if len(X) == 0:
        debug_scan(args.data_dir)
        raise RuntimeError(f"[Data] No training samples loaded from {args.data_dir}. "
                           "Please check file patterns or use --dry_run to inspect.")

    dset = TensorDataset(X, Y_in, Y_out)
    dl = DataLoader(
        dset, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=min(8, os.cpu_count() or 4),
        pin_memory=True, persistent_workers=True
    )

    dlv = None
    if args.valid_dir:
        Xv, Yvin, Yvout = load_dataset(args.valid_dir, stoi, args.max_len, limit=args.valid_limit)
        if len(Xv) == 0:
            debug_scan(args.valid_dir)
            print(f"[Warn] No validation samples loaded from {args.valid_dir}. Metrics will be skipped.")
            dlv = None
        else:
            dsetv = TensorDataset(Xv, Yvin, Yvout)
            dlv = DataLoader(
                dsetv, batch_size=args.batch_size, shuffle=False,
                num_workers=min(8, os.cpu_count() or 4),
                pin_memory=True, persistent_workers=True
            )

    # ---- 模型 ----
    model = TransformerSeq2Seq(
        src_vocab=len(itos), tgt_vocab=len(itos),
        d_model=args.d_model, n_heads=(1 if args.ablate=='single_head' else args.n_heads),
        num_layers=args.num_layers, ff_dim=args.ff_dim, dropout=args.dropout,
        max_len=args.max_len, share_embeddings=args.share_embeddings,
        use_posenc=(False if args.ablate=='no_posenc' else True)
    ).to(device)

    if not args.no_compile:
        try:
            model = torch.compile(model, mode="max-autotune")
        except Exception as e:
            print(f"[warn] torch.compile failed: {e}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params/1e6:.2f}M")

    # ---- 优化器 & 调度 ----
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9,0.98),
        eps=1e-9, weight_decay=args.weight_decay
    )
    scheduler = InverseSqrtWithWarmup(optimizer, warmup_steps=args.warmup_steps)
    ckpt = Checkpointer(args.ckpt_dir)

    # ---- AMP/累积梯度策略 ----
    # fp16 需要 GradScaler；bf16/ fp32 不需要
    use_fp16_scaler = (args.amp_dtype == 'fp16') and (not args.no_amp)
    scaler = torch.amp.GradScaler('cuda',enabled=use_fp16_scaler)

    amp_enabled = (args.amp_dtype in ['bf16','fp16']) and (not args.no_amp)
    amp_dtype = {
        'bf16': torch.bfloat16,
        'fp16': torch.float16,
        'fp32': torch.float32
    }[args.amp_dtype]

    # ---- 损失 ----
    crit = nn.CrossEntropyLoss(
        ignore_index=PAD,
        label_smoothing=(0.0 if args.ablate == 'no_label_smoothing' else 0.1)
    )

    history = {'train_loss':[], 'lrs':[]}
    best_bleu = -1.0

    # ---- 训练循环（含梯度累积，正确的 step/zero_grad 顺序）----
    for epoch in range(1, args.epochs+1):
        model.train()
        loss_meter = SmoothedValue(50)

        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(dl, desc=f'epoch {epoch}', ncols=100)
        for step, (xb, y_in, y_out) in enumerate(pbar, start=1):
            xb   = xb.to(device, non_blocking=True)
            y_in = y_in.to(device, non_blocking=True)
            y_out= y_out.to(device, non_blocking=True)

            with torch.amp.autocast('cuda', enabled=amp_enabled, dtype=amp_dtype):
                logits = model(
                    xb, y_in,
                    src_key_padding_mask=(xb!=PAD),
                    tgt_key_padding_mask=(y_in!=PAD)
                )
                loss = crit(logits.view(-1, logits.size(-1)), y_out.view(-1))
                # 梯度累积：把 loss 均分到每个小步
                loss = loss / max(1, args.grad_accum)

            if use_fp16_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # 每 grad_accum 小步更新一次权重
            if step % max(1, args.grad_accum) == 0:
                if use_fp16_scaler:
                    scaler.unscale_(optimizer)
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                if use_fp16_scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad(set_to_none=True)  # 下一次累积前清梯度
                scheduler.step()

            loss_meter.update(loss.item() * max(1, args.grad_accum))  # 还原到未除的 loss 以便观测
            history['train_loss'].append(loss.item() * max(1, args.grad_accum))
            history['lrs'].append(optimizer.param_groups[0]['lr'])
            pbar.set_postfix({
                "loss": f"{loss_meter.median:.3f}",
                "lr":   f"{optimizer.param_groups[0]['lr']:.2e}"
            })

        # ---- 保存 last ----
        ckpt.save({
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'stoi': stoi, 'itos': itos,
            'args': vars(args),
            'epoch': epoch
        }, name='last.pt')

        # ---- 验证 ----
        if dlv is not None:
            if args.valid_dir and os.path.exists(os.path.join(args.valid_dir, "input.txt")):
                bleu, rouge = run_eval_ducref(model, args.valid_dir, stoi, itos, device, args)
            else:
                bleu, rouge = run_eval(model, dlv, itos, device, args.max_len)
            print(f"Eval BLEU-4: {bleu:.2f} | ROUGE-1(F1): {rouge['rouge1_f1']:.2f} | ROUGE-L(F1): {rouge['rougeL_f1']:.2f}")
            if bleu > best_bleu:
                best_bleu = bleu
                ckpt.save({'model': model.state_dict(), 'itos': itos, 'stoi': stoi, 'args': vars(args)}, name='best.pt')

        # ---- 曲线 ----
        plot_curves(history, os.path.join(args.work_dir, 'plots', 'train.png'))

def evaluate(args):
    # 在 train(args) 和 evaluate(args) 里调用：
    args.data_dir = str(resolve_under_root(args.data_dir))
    args.valid_dir = str(resolve_under_root(args.valid_dir)) if args.valid_dir else None
    args.work_dir = str(resolve_under_root(args.work_dir))
    args.ckpt_dir = str(resolve_under_root(args.ckpt_dir))
    args.ckpt_path = str(resolve_under_root(args.ckpt_path))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    obj = torch.load(args.ckpt_path, map_location='cpu')
    itos = obj['itos']; stoi = obj['stoi']
    model_args = obj.get('args', {})
    model = TransformerSeq2Seq(
        src_vocab=len(itos), tgt_vocab=len(itos),
        d_model=model_args.get('d_model',256),
        n_heads=model_args.get('n_heads',4),
        num_layers=model_args.get('num_layers',4),
        ff_dim=model_args.get('ff_dim',1024),
        dropout=model_args.get('dropout',0.1),
        max_len=args.max_len,
        share_embeddings=model_args.get('share_embeddings',False),
        use_posenc=model_args.get('use_posenc',True)
    ).to(device)
    model.load_state_dict(obj['model'])

    X, Y_in, Y_out = load_dataset(args.data_dir, stoi, args.max_len, limit=args.valid_limit)
    dl = DataLoader(TensorDataset(X,Y_in,Y_out), batch_size=args.batch_size, shuffle=False)
    bleu, rouge = run_eval(model, dl, itos, device, args.max_len)
    print(json.dumps({'BLEU4': bleu, **rouge}, ensure_ascii=False, indent=2))

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--mode', default='train', choices=['train','eval'])
    p.add_argument('--data_dir', required=True)
    p.add_argument('--valid_dir', default=None)
    p.add_argument('--work_dir', default='outputs/exp1')
    p.add_argument('--ckpt_dir', default='checkpoints')
    p.add_argument('--ckpt_path', default='checkpoints/best.pt')
    p.add_argument('--vocab_size', type=int, default=32000)
    p.add_argument('--max_len', type=int, default=128)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--d_model', type=int, default=256)
    p.add_argument('--n_heads', type=int, default=4)
    p.add_argument('--num_layers', type=int, default=4)
    p.add_argument('--ff_dim', type=int, default=1024)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--share_embeddings', action='store_true')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight_decay', type=float, default=0.0)
    p.add_argument('--warmup_steps', type=int, default=4000)
    p.add_argument('--max_grad_norm', type=float, default=1.0)
    p.add_argument('--no_amp', action='store_true')
    p.add_argument('--train_limit', type=int, default=None)
    p.add_argument('--valid_limit', type=int, default=None)
    p.add_argument('--ablate', default='none', choices=['none','no_posenc','single_head','no_label_smoothing'])
    p.add_argument('--dry_run', action='store_true')
    p.add_argument('--rebuild_vocab', action='store_true',
                   help='Force rebuild vocabulary even if vocab.json exists')
    p.add_argument('--min_vocab_size', type=int, default=100,
                   help='If existing vocab smaller than this, rebuild it')
    p.add_argument('--no_compile', action='store_true')
    p.add_argument('--grad_accum', type=int, default=1)
    p.add_argument('--amp_dtype', default='bf16', choices=['bf16', 'fp16', 'fp32'])
    p.add_argument('--min_gen_len', type=int, default=5,
                   help='minimum generation length before allowing EOS')
    p.add_argument('--decode', default='beam', choices=['greedy', 'beam'])
    p.add_argument('--beam_size', type=int, default=4)
    p.add_argument('--length_penalty', type=float, default=0.6)
    p.add_argument('--no_repeat_ngram_size', type=int, default=0)
    p.add_argument("--eos_bias", type=float, default=2.0,
                        help="Bias added to EOS token during beam search; higher = earlier stop")
    return p.parse_args()

if __name__=='__main__':
    args = parse_args()

    from pathlib import Path

    PROJ_ROOT = Path(__file__).resolve().parents[1]


    def resolve_under_root(p):
        if p is None:
            return None
        p = Path(p)
        return p if p.is_absolute() else (PROJ_ROOT / p)


    # 在 train(args) 和 evaluate(args) 里调用：
    # args.data_dir = str(resolve_under_root(args.data_dir))
    # args.valid_dir = str(resolve_under_root(args.valid_dir)) if args.valid_dir else None
    # args.work_dir = str(resolve_under_root(args.work_dir))
    # args.ckpt_dir = str(resolve_under_root(args.ckpt_dir))
    # args.ckpt_path = str(resolve_under_root(args.ckpt_path))

    if args.mode=='train':
        train(args)
    else:
        evaluate(args)
