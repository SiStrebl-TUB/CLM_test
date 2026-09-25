"""Tables and the decision from the pre-check outputs.

Finds every run.py JSON below the given folders (also when a copy from the cluster landed a folder deeper) and
prints, per run: top-1 and pairwise win rate per condition, scorer and kind, CLM's rates per mutation operator
and action type, and the decision fixed in the README, computed from <tag>_items.jsonl.gz and the labelled
<tag>_audit.csv next to the JSON:

  cell      the budget x format with the highest CLM pairwise win rate on neighbour sets
  1 sanity  CLM top-1 on random >= 0.8
  2 main    CLM pairwise win rate on mutations (per negative), against raw and lexical
  3 ceiling q = share of labelled negatives as good as the true action (unclear = 1/2); ceiling 1 - q/2;
            operators with q > 0.3 on >= 10 labels are excluded (and to be fixed before the A/B test)
  4 verdict headroom = ceiling - CLM: > 5 pp GO, < 5 pp STOP, 95 % interval containing 5 pp: GREY ZONE
            (add the reference scorer first)

    python precheck/analyze.py                  # all runs under exps/
    python precheck/analyze.py exps/precheck/v1.json
"""
import glob, gzip, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from precheck.audit import read_labels
from precheck.run import boot_draws

SANITY_TOP1, HEADROOM, Q_FLAG, Q_FLAG_MIN_N, N_BOOT = 0.80, 0.05, 0.30, 10, 2000     # the rule in the README


def load(paths):
    files = []
    for p in paths or ["exps"]:
        files += [p] if p.endswith(".json") else glob.glob(os.path.join(p, "**", "*.json"), recursive=True)
    out = []
    for f in sorted(set(files)):
        try: d = json.load(open(f))
        except (ValueError, OSError): continue
        if isinstance(d, dict) and d.get("kind") == "clm_precheck": out.append((f, d))
    return out


def ci(m, c): return "      -       " if m is None else f"{m:.3f} [{c[0]:.2f},{c[1]:.2f}]"
def ci95(draws): return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]
def iv(c): return f"[{c[0]:.3f}, {c[1]:.3f}]"


def cell(res, sc, kd, L, fm):
    return next((r for r in res if r["scorer"] == sc and r["kind"] == kd and r["format"] == fm and r["max_len"] in (L, 0)), None)


def report(f, d):
    cfg, res = d["config"], d["results"]
    print(f"\n##### {f}  (tag {d['tag']}, {d['counts']['items']} steps / {d['counts']['instances']} instances, k={cfg['k']}, "
          f"{cfg.get('gpu', 'no GPU')}, CLM {str(cfg.get('clm_commit', '?'))[:7]}{', DRY RUN' if cfg.get('dry_run') else ''})")
    print(f"full candidate sets: {d['counts']['full_sets']}   truncated states: {d['counts'].get('truncated_share')}")
    conds = sorted({(r["max_len"], r["format"]) for r in res if r["max_len"]}, key=lambda c: (-c[0], c[1] != "turn"))
    print(f"\ntop-1 among 1+{cfg['k']} (chance {res[0]['chance']:.3f}) | pairwise win rate per negative, 95 % interval over instances")
    print(f"{'scorer':8s} {'kind':10s} " + " ".join(f"{f'{L}/{fm}':>33s}" for L, fm in conds))
    for sc in ("clm", "raw", "lexical"):
        for kd in ("random", "neighbour", "mutation"):
            cells = []
            for L, fm in conds:
                r = cell(res, sc, kd, L, fm)
                cells.append(f"{ci(r['top1'], r['top1_ci']):>16s} | {r['pairwise']:.3f}" if r else " " * 33)
            print(f"{sc:8s} {kd:10s} " + " ".join(f"{c:>33s}" for c in cells))
    L, fm = conds[0]
    for kd in ("mutation", "neighbour"):
        r = cell(res, "clm", kd, L, fm)
        if not r: continue
        print(f"\nclm {kd}, {L}/{fm}: pairwise win rate per {'operator' if kd == 'mutation' else 'position'}")
        for op, v in sorted(r["by_op"].items(), key=lambda kv: kv[1]["pairwise"]):
            print(f"  {op:24s} n={v['n']:5d}  {ci(v['pairwise'], v['ci'])}")
        print(f"clm {kd}, {L}/{fm}: by action (pairwise, n negatives / top-1, n steps)")
        for sub, v in r["by_sub"]["pairwise"].items():
            t = r["by_sub"]["top1"].get(sub)
            print(f"  {sub:24s} {v['mean']:.3f}  n={v['n']:5d} / " + (f"{t['mean']:.3f}  n={t['n']}" if t else "-"))
        if r.get("ece") is not None: print(f"  ECE of the top probability (full sets): {r['ece']:.3f}, mean p(true) {r['p_true']:.3f}")


# ----------------------------------------------------------------------------- decision
def pairs(items, key, kind="mutation"):
    """(instance, sid, candidate id, operator, win) of the true action against each negative; win 1, 1/2 on a tie, 0"""
    out = []
    for it in items:
        st = it["sets"][kind]; s = st["scores"].get(key) or []
        out += [(it["instance"], it["sid"], f"m{j}", st["ops"][j], 1.0 if s[0] > s[j] else 0.5 if s[0] == s[j] else 0.0) for j in range(1, len(s))]
    return out


def kappa(a, b):
    """Cohen's kappa of two label lists"""
    n, cats = len(a), set(a) | set(b)
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum(a.count(c) * b.count(c) for c in cats) / n ** 2
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def decide(f, d):
    res = d["results"]
    nb = {(r["max_len"], r["format"]): r["pairwise"] for r in res if r["scorer"] == "clm" and r["kind"] == "neighbour"}
    L, fm = max(nb, key=nb.get)
    print(f"\n=== decision (rule in the README), cell {L}/{fm}: highest CLM pairwise on neighbour sets "
          f"({', '.join(f'{a}/{b} {v:.3f}' for (a, b), v in sorted(nb.items(), key=lambda kv: -kv[1]))})")
    san = cell(res, "clm", "random", L, fm)["top1"]
    ok = san is not None and san >= SANITY_TOP1
    print(f"1 sanity    CLM top-1 on random {san if san is None else round(san, 3)} (needs >= {SANITY_TOP1})  {'ok' if ok else 'FAILED'}")
    ip = f[:-5] + "_items.jsonl.gz"
    if not os.path.exists(ip):
        print(f"            missing {ip}: copy it next to the JSON\n=> no verdict yet"); return None
    items = [json.loads(line) for line in gzip.open(ip, "rt")]
    inst = {it["sid"]: it["instance"] for it in items}
    P = pairs(items, f"{L}|{fm}|clm")
    pw, pw_d = boot_draws([p[4] for p in P], [p[0] for p in P], N_BOOT, 1)
    base = {sc: float(np.mean([p[4] for p in pairs(items, k)])) for sc, k in (("raw", f"{L}|{fm}|raw"), ("lexical", f"0|{fm}|lexical"))}
    print(f"2 main      CLM pairwise on mutations {pw:.3f} {iv(ci95(pw_d))}   raw {base['raw']:.3f}, lexical {base['lexical']:.3f}"
          + ("" if pw > max(base.values()) else "   <- the head does not beat the baselines"))
    ap = f[:-5] + "_audit.csv"
    lab = read_labels(ap) if os.path.exists(ap) else {}
    if not lab:
        print(f"3 ceiling   no labels yet: label {ap} (see precheck/audit.py), then rerun\n=> no verdict yet"); return None
    by_op = {}
    for (op, v) in lab.values(): by_op.setdefault(op, []).append(v)
    flagged = {op: (float(np.mean(v)), len(v)) for op, v in by_op.items() if len(v) >= Q_FLAG_MIN_N and np.mean(v) > Q_FLAG}
    keep = [(inst.get(sid, sid), v) for (sid, _), (op, v) in lab.items() if op not in flagged]
    q, q_d = boot_draws([v for _, v in keep], [g for g, _ in keep], N_BOOT, 2)
    n_lab = {name: sum(v == x for _, v in lab.values()) for name, x in (("same", 1.0), ("unclear", 0.5), ("worse", 0.0))}
    print(f"3 ceiling   {len({s for s, _ in lab})} sets, {len(lab)} negatives labelled {n_lab}; q = {q:.3f} -> ceiling 1 - q/2 = "
          f"{1 - q / 2:.3f} {iv(ci95(1 - q_d / 2))}")
    for op, (qo, n) in sorted(flagged.items()):
        print(f"            {op}: q {qo:.2f} on {n} labels > {Q_FLAG} -> excluded here; fix this operator before the A/B test")
    lb = read_labels(ap, "label_b"); both = [k for k in lb if k in lab]
    if both:
        x, y = [lab[k][1] for k in both], [lb[k][1] for k in both]
        print(f"            cross-check label_b: {len(both)} negatives, agreement {np.mean([u == w for u, w in zip(x, y)]):.2f}, kappa {kappa(x, y):.2f}")
    Pk = [p for p in P if p[3] not in flagged]
    pwk, pwk_d = boot_draws([p[4] for p in Pk], [p[0] for p in Pk], N_BOOT, 3)
    h, h_d = (1 - q / 2) - pwk, (1 - q_d / 2) - pwk_d
    lo, hi = ci95(h_d)
    clean = [p[4] for p in Pk if lab.get((p[1], p[2]), (None, None))[1] == 0.0]
    print(f"4 headroom  ceiling - CLM = {h:.3f} {iv([lo, hi])}" + (f"   (check: 1 - CLM pairwise on the {len(clean)} negatives labelled worse = {1 - np.mean(clean):.3f})" if clean else ""))
    if not ok: verdict = "STOP: the sanity check failed, check the rendering of states and actions first"
    elif lo <= HEADROOM <= hi: verdict = "GREY ZONE: the interval contains 5 pp, add the reference scorer (Qwen3-8B likelihood) before deciding"
    elif h > HEADROOM: verdict = "GO: more than 5 pp headroom, run the A/B training experiment"
    else: verdict = "STOP: CLM is within 5 pp of the ceiling, hard negatives for actions are not the lever"
    print(f"=> {verdict}")
    return verdict.split(":")[0]


if __name__ == "__main__":
    runs = load(sys.argv[1:])
    if not runs: sys.exit("no pre-check JSON found")
    for f, d in runs: report(f, d); decide(f, d)
