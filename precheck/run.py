"""Pre-check: does the released CLM head tell the action an agent took from near misses?

CLM-8B (frozen Qwen3-8B, last-token pooling, trained projection heads) is scored on agent steps it has not been
trained on: resolved SWE-rebench OpenHands trajectories, which are not among the ADP sources of its
post-training. For every sampled step the taken action competes against k negatives of one kind at a time:

  random     true actions of the same tool from other tasks -- the sanity check: must be easy, else the format is off
  neighbour  other actions of the same trajectory, nearest steps first -- what CLM's post-training separates
  mutation   near misses of the true call (precheck/mutate.py) -- what hard negatives for actions would add

Two unknowns of CLM's recipe are crossed as conditions: the state budget (2048 tokens as served, 8192 as in
fine-tuning; states are cut from the front) and the action format ("turn" = thought + call, "call" = call alone).
Scorers: the CLM head (cosine of the projections, times its logit scale for probabilities), the raw encoder
(cosine of the Qwen3 embeddings: CLM's clm-raw ablation) and a lexical baseline (the share of the action's tokens
that occur in the last 8000 characters of the state; independent of the budget, stored with max_len 0).

Per condition, scorer and kind: top-1 among 1 + k (full sets only, chance 1/(k+1)), the pairwise win rate of the
true action against each negative (ties count 1/2), MRR, the softmax probability of the true action, the ECE of
the top probability, and win rates per operator / neighbour position. 95 % intervals from a bootstrap over
instances (the unit that is sampled).

Tokens, embeddings and heads come from CLM's own code (external/CLM at the pinned commit): states through
Recipe.state_ids (chat template, last max_len - 1 tokens), actions through Recipe.text_ids(keep="head"), the
encoder through OfflineBackend (in-process vLLM pooling), the heads through clm.heads.HeadPair as in clm-serve.
The recipe is run once without a cap and cut here; the first items are checked against Recipe(max_len) itself.

    python precheck/run.py --rows-json sample_rows.json --dry-run --n-states 40 --tag smoke    # local, no GPU
    python precheck/run.py --parquet exps/precheck/data/trajectories.parquet --n-states 2000 --tag v1
"""
import argparse, gzip, hashlib, json, math, os, random, re, subprocess, sys, time, zlib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from precheck.data import FORMATS, TOOLS, category, download, neighbours, random_negatives, read_parquet, read_rows_json, render_action, state_blob, state_messages, steps_of
from precheck.mutate import context, mutations

CLM_DIR = os.path.join(ROOT, "external", "CLM")
KINDS = ("random", "neighbour", "mutation")
RAW_SCALE = 100.0                                   # clm-raw's scale in clm.engine; only the probabilities depend on it
LEX_RE = re.compile(r"[A-Za-z_][\w./-]{2,}")
CHUNK = 8192


# ----------------------------------------------------------------------------- stand-ins for --dry-run
class FakeRecipe:
    """embed_utils.Recipe without a tokenizer: about 4 characters per token, ids from the characters"""
    def text_ids(self, text, keep="head"):
        return [zlib.crc32(text[i:i + 4].encode()) % 150000 for i in range(0, len(text), 4)] or [0]

    def state_ids(self, msgs):
        return self.text_ids("".join(m["content"] + "".join(t["function"]["arguments"] for t in m.get("tool_calls", [])) for m in msgs))


class FakeBackend:
    """the encoder replaced by a random unit vector per distinct input: every scorer but lexical is at chance"""
    def __init__(self, dim=64): self.dim = dim

    def embed(self, id_lists):
        x = np.stack([np.random.default_rng(zlib.crc32(np.asarray(ids, np.int64).tobytes())).standard_normal(self.dim) for ids in id_lists])
        return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


class FakeHead:
    scale = 20.0

    def __init__(self, d):
        r = np.random.default_rng(0); self.Ws, self.Wa = r.standard_normal((d, 32)), r.standard_normal((d, 32))

    def project_states(self, x): z = x.astype(np.float32) @ self.Ws; return z / np.linalg.norm(z, axis=1, keepdims=True)
    def project_actions(self, x): z = x.astype(np.float32) @ self.Wa; return z / np.linalg.norm(z, axis=1, keepdims=True)


class Head:
    """clm.heads.HeadPair on the CPU (the GPU stays with vLLM), in chunks, returning numpy"""
    def __init__(self, path):
        from clm.heads import HeadPair
        self.hp = HeadPair("clm-latest", path, device="cpu").ensure(); self.scale = self.hp.scale

    def _run(self, f, x): return np.concatenate([f(x[i:i + CHUNK].astype(np.float32)).numpy() for i in range(0, len(x), CHUNK)])
    def project_states(self, x): return self._run(self.hp.project_states, x)
    def project_actions(self, x): return self._run(self.hp.project_actions, x)


# ----------------------------------------------------------------------------- items
def build_items(rows, args):
    """sampled steps with their state and the three candidate sets"""
    trajs = [(r, s) for r in rows if len(s := steps_of(r["trajectory"])) > args.k]
    pool = {t: [] for t in TOOLS}
    for r, s in trajs:
        for _, a in s: pool[a["name"]].append((r["instance_id"], a))
    rng, items = random.Random(args.seed), []
    for r, s in trajs:
        for j in sorted(rng.sample(range(len(s)), min(args.steps_per_traj, len(s)))):
            i, a = s[j]; sid = f"{r['trajectory_id']}:{i}"
            msgs = state_messages(r["trajectory"], i); blob = state_blob(msgs)
            negs = {"random": random_negatives(pool, a, r["instance_id"], args.k, random.Random(sid)),
                    "neighbour": neighbours(s, j, args.k), "mutation": mutations(a, context(blob), args.k, seed=sid)}
            sub = f"editor.{a['args'].get('command')}" if a["name"] == "str_replace_editor" else f"bash.{category(a)}"
            items.append(dict(sid=sid, instance=r["instance_id"], tool=a["name"], sub=sub, category=category(a), msg_idx=i,
                              action=a, cands={kd: [("true", a)] + negs[kd] for kd in KINDS}, msgs=msgs, blob=blob))
    return items[:args.n_states]


def embed_all(items, args, recipe, backend):
    """CLM's token recipe for every state (per budget) and candidate (per budget and format); each distinct token
    list is embedded once. Returns the embeddings and sets it["n_tokens"], it["srow"], it["crow"]"""
    uniq, tok = {}, {}
    row = lambda ids: uniq.setdefault(tuple(ids), len(uniq))
    for it in items:
        full = recipe.state_ids(it["msgs"]); it["n_tokens"] = len(full)
        it["srow"] = {L: row(full[-(L - 1):]) for L in args.max_lens}
        it["crow"] = {}
        for fmt in args.formats:
            for kd, cands in it["cands"].items():
                ids = []
                for _, a in cands:
                    t = render_action(a, fmt)
                    if t not in tok: tok[t] = recipe.text_ids(t, keep="head")
                    ids.append(tok[t])
                for L in args.max_lens: it["crow"][(L, fmt, kd)] = [row(x[:L - 1]) for x in ids]
    keys = list(uniq)
    print(f"[embed] {len(keys)} distinct inputs, {sum(map(len, keys)) / 1e6:.1f}M tokens", flush=True)
    return np.concatenate([backend.embed([list(k) for k in keys[i:i + CHUNK]]).astype(np.float16) for i in range(0, len(keys), CHUNK)])


def score_all(items, E, head, args):
    """{(budget, format, scorer, kind): [candidate scores per item, true action first]}"""
    Zs, Za, Er = head.project_states(E), head.project_actions(E), E.astype(np.float32)
    out = {}
    for it in items:
        for (L, fmt, kd), rows in it["crow"].items():
            s = it["srow"][L]
            out.setdefault((L, fmt, "clm", kd), []).append(Za[rows] @ Zs[s])
            out.setdefault((L, fmt, "raw", kd), []).append(Er[rows] @ Er[s])
        S = set(LEX_RE.findall(it["blob"][-8000:].lower()))
        for fmt in args.formats:
            for kd, cands in it["cands"].items():
                A = [set(LEX_RE.findall(render_action(a, fmt).lower())) for _, a in cands]
                out.setdefault((0, fmt, "lexical", kd), []).append(np.array([len(x & S) / max(1, len(x)) for x in A]))
    return out


# ----------------------------------------------------------------------------- metrics
def softmax(x):
    e = np.exp(x - x.max()); return e / e.sum()


def boot(values, groups, n_boot, seed=0):
    """mean and 95 % interval, resampling instances"""
    v = np.asarray(values, float)
    if len(v) == 0: return None, None
    _, inv = np.unique(np.asarray(groups), return_inverse=True)
    sums, cnts = np.bincount(inv, weights=v), np.bincount(inv)
    idx = np.random.default_rng(seed).integers(0, len(sums), size=(n_boot, len(sums)))
    st = sums[idx].sum(1) / cnts[idx].sum(1)
    return float(v.mean()), [float(np.percentile(st, 2.5)), float(np.percentile(st, 97.5))]


def ece(conf, correct, bins=10):
    c, y = np.asarray(conf), np.asarray(correct)
    b = np.minimum((c * bins).astype(int), bins - 1)
    return float(sum(abs(y[b == i].mean() - c[b == i].mean()) * (b == i).mean() for i in range(bins) if (b == i).any()))


def by(recs, key, field):
    d = {}
    for r in recs:
        if r[field] is not None: d.setdefault(r[key], []).append(r[field])
    return {k: {"mean": float(np.mean(v)), "n": len(v)} for k, v in sorted(d.items(), key=lambda kv: str(kv[0]))}


def summarize(items, scores, scales, args):
    res = []
    for (L, fmt, scorer, kd), S in sorted(scores.items(), key=lambda kv: (kv[0][:3], KINDS.index(kv[0][3]))):
        recs, ops = [], {}
        for it, s in zip(items, S):
            if len(s) < 2: continue
            s0, sn = s[0], s[1:]
            wins = (s0 > sn) + 0.5 * (s0 == sn)
            full = len(sn) == args.k
            top = (1.0 if s0 > sn.max() else 1.0 / (1 + (sn == s0).sum()) if s0 == sn.max() else 0.0) if full else None
            p = softmax(scales[scorer] * s) if scorer in scales else None
            recs.append(dict(g=it["instance"], pw=wins.mean(), rr=1.0 / (1 + (sn > s0).sum() + 0.5 * (sn == s0).sum()), top=top,
                             p=None if p is None else p[0], conf=None if p is None or not full else p.max(), cat=it["category"],
                             sub=it["sub"], trunc=("truncated" if it["n_tokens"] > L - 1 else "complete") if L else "n/a"))
            for (op, _), w in zip(it["cands"][kd][1:], wins): ops.setdefault(op, ([], []))[0].append(w); ops[op][1].append(it["instance"])
        tops = [r for r in recs if r["top"] is not None]; confs = [r for r in tops if r["conf"] is not None]
        pw, pw_ci = boot([r["pw"] for r in recs], [r["g"] for r in recs], args.n_boot)
        t1, t1_ci = boot([r["top"] for r in tops], [r["g"] for r in tops], args.n_boot)
        res.append(dict(max_len=L, format=fmt, scorer=scorer, kind=kd, n=len(recs), n_full=len(tops), chance=1.0 / (args.k + 1),
                        top1=t1, top1_ci=t1_ci, pairwise=pw, pairwise_ci=pw_ci, mrr=float(np.mean([r["rr"] for r in recs])) if recs else None,
                        p_true=float(np.mean([r["p"] for r in recs])) if recs and recs[0]["p"] is not None else None,
                        ece=ece([r["conf"] for r in confs], [r["top"] for r in confs]) if confs else None,
                        by_category={"pairwise": by(recs, "cat", "pw"), "top1": by(recs, "cat", "top")},
                        by_sub={"pairwise": by(recs, "sub", "pw"), "top1": by(recs, "sub", "top")},
                        by_truncation={"pairwise": by(recs, "trunc", "pw"), "top1": by(recs, "trunc", "top")},
                        by_op={op: dict(zip(("pairwise", "ci"), boot(v, g, args.n_boot)), n=len(v)) for op, (v, g) in sorted(ops.items())}))
    return res


# ----------------------------------------------------------------------------- output
def fmt_ci(m, ci): return "   -   " if m is None else f"{m:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]"


def print_table(res, L, fmt):
    print(f"\n=== budget {L}, format {fmt}: top-1 among 1+k (chance {res[0]['chance']:.3f}) | pairwise win rate", flush=True)
    for r in res:
        if r["format"] == fmt and r["max_len"] in (L, 0):
            print(f"  {r['scorer']:8s} {r['kind']:10s} n={r['n']:5d} full={r['n_full']:5d}  top1 {fmt_ci(r['top1'], r['top1_ci'])}  pw {fmt_ci(r['pairwise'], r['pairwise_ci'])}")
    clm = [r for r in res if (r["max_len"], r["format"], r["scorer"], r["kind"]) == (L, fmt, "clm", "mutation")]
    if clm:
        print("  clm, pairwise win rate per mutation operator:")
        for op, d in sorted(clm[0]["by_op"].items(), key=lambda kv: kv[1]["pairwise"]):
            print(f"    {op:24s} n={d['n']:5d}  {fmt_ci(d['pairwise'], d['ci'])}")


def write_examples(path, items, scores, key, n=20, seed=0):
    """a readable sample for checking the negatives by hand (are some mutations just as right?)"""
    lines = [f"# Candidate sets, scores from {key}\n"]
    for idx in sorted(random.Random(seed).sample(range(len(items)), min(n, len(items)))):
        it = items[idx]
        lines += [f"## {it['sid']}  ({it['sub']}, {it['n_tokens']} state tokens)\n", "State, last 1200 characters:\n", "```text", it["blob"][-1200:], "```\n"]
        for kd in ("mutation", "neighbour"):
            s = scores[(*key, kd)][idx]
            lines.append(f"**{kd}** (true action first)\n")
            for (op, a), v in list(zip(it["cands"][kd], s))[: (len(s) if kd == "mutation" else 4)]:
                lines.append(f"- `{op}` {v:+.3f}\n  ```text\n  " + render_action(a, "call")[:500].replace("\n", "\n  ") + "\n  ```")
            lines.append("")
    open(path, "w").write("\n".join(lines))


def versions():
    out = {}
    for mod in ("vllm", "transformers", "torch", "numpy"):
        try: out[mod] = __import__(mod).__version__
        except Exception: pass
    try:
        import torch
        if torch.cuda.is_available(): out["gpu"] = torch.cuda.get_device_name(0)
    except Exception: pass
    try: out["clm_commit"] = subprocess.run(["git", "-C", CLM_DIR, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception: pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=None, help="trajectories.parquet; downloaded to exps/precheck/data if neither source is given")
    ap.add_argument("--rows-json", default=None, help="rows as JSON (datasets-server /rows output), for local tests")
    ap.add_argument("--n-states", type=int, default=2000); ap.add_argument("--steps-per-traj", type=int, default=2); ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--max-lens", type=int, nargs="+", default=[2048, 8192]); ap.add_argument("--formats", nargs="+", default=list(FORMATS), choices=FORMATS)
    ap.add_argument("--all-trajectories", action="store_true", help="also unresolved trajectories")
    ap.add_argument("--model", default="Qwen/Qwen3-8B"); ap.add_argument("--ckpt", default=None, help="head checkpoint; default: the released CLM_v0.1-8B.pt")
    ap.add_argument("--gpu-mem", type=float, default=0.85); ap.add_argument("--n-boot", type=int, default=1000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="stand-ins for tokenizer, encoder and head: tests the pipeline without a GPU")
    ap.add_argument("--fake-encoder", action="store_true", help="real tokenizer and head, random encoder: tests everything but vLLM, on a CPU")
    ap.add_argument("--tag", default="v1"); ap.add_argument("--out-dir", default="exps/precheck")
    args = ap.parse_args()
    t0 = time.time(); os.makedirs(args.out_dir, exist_ok=True)

    n_traj = math.ceil(1.1 * args.n_states / args.steps_per_traj)          # a few spare: short trajectories are dropped
    rows = (read_rows_json(args.rows_json, n_traj, args.seed, not args.all_trajectories) if args.rows_json else
            read_parquet(args.parquet or download(), n_traj, args.seed, not args.all_trajectories))
    items = build_items(rows, args); del rows
    counts = {"items": len(items), "instances": len({it["instance"] for it in items}),
              "full_sets": {kd: sum(len(it["cands"][kd]) == args.k + 1 for it in items) for kd in KINDS},
              "tool": {t: sum(it["tool"] == t for it in items) for t in TOOLS},
              "category": {c: sum(it["category"] == c for it in items) for c in sorted({it["category"] for it in items})}}
    print(f"[items] {json.dumps(counts)}  ({time.time() - t0:.0f}s)", flush=True)

    if args.dry_run:
        recipe, backend, head, ckpt = FakeRecipe(), FakeBackend(64), FakeHead(64), None
    else:
        sys.path[:0] = [os.path.join(CLM_DIR, "src"), os.path.join(CLM_DIR, "train")]
        import embed_utils
        from clm.heads import download as download_head
        recipe = embed_utils.Recipe(args.model, max_len=10**9 + 1)        # uncapped; cut per budget in embed_all
        for L in args.max_lens:
            ref = embed_utils.Recipe(args.model, max_len=L)
            for it in items[:3]:
                assert ref.state_ids(it["msgs"]) == recipe.state_ids(it["msgs"])[-(L - 1):], "cutting the uncapped recipe differs from Recipe(max_len)"
                t = render_action(it["action"], args.formats[0])
                assert ref.text_ids(t, keep="head") == recipe.text_ids(t, keep="head")[:L - 1], "same for actions"
        ckpt = args.ckpt or download_head()
        head = Head(ckpt)
        backend = FakeBackend(4096) if args.fake_encoder else embed_utils.OfflineBackend(args.model, max(args.max_lens), args.gpu_mem)
    E = embed_all(items, args, recipe, backend)
    print(f"[embed] done ({time.time() - t0:.0f}s)", flush=True)
    counts["truncated_share"] = {L: float(np.mean([it["n_tokens"] > L - 1 for it in items])) for L in args.max_lens}
    scores = score_all(items, E, head, args)
    res = summarize(items, scores, {"clm": head.scale, "raw": RAW_SCALE}, args)

    tag = os.path.join(args.out_dir, args.tag)
    config = {**vars(args), **versions(), "head_sha256": hashlib.sha256(open(ckpt, "rb").read()).hexdigest() if ckpt else None,
              "head_scale": float(head.scale), "minutes": round((time.time() - t0) / 60, 1)}
    json.dump({"kind": "clm_precheck", "tag": args.tag, "config": config, "counts": counts, "results": res}, open(tag + ".json", "w"), indent=1)
    with gzip.open(tag + "_items.jsonl.gz", "wt") as f:
        for idx, it in enumerate(items):
            sets = {kd: {"ops": [op for op, _ in it["cands"][kd]],
                         "scores": {f"{L}|{fmt}|{sc}": [round(float(v), 5) for v in S[idx]] for (L, fmt, sc, k2), S in scores.items() if k2 == kd}} for kd in KINDS}
            f.write(json.dumps({k: it[k] for k in ("sid", "instance", "tool", "sub", "category", "msg_idx", "n_tokens")} | {"sets": sets}) + "\n")
    primary = (max(args.max_lens), args.formats[0], "clm")
    write_examples(tag + "_examples.md", items, scores, primary)
    for L in sorted(args.max_lens, reverse=True):
        for fmt in args.formats: print_table(res, L, fmt)
    print(f"\nwritten {tag}.json, {tag}_items.jsonl.gz, {tag}_examples.md  ({config['minutes']} min)")


if __name__ == "__main__":
    main()
