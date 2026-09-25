"""Tables from the pre-check JSONs.

Finds every run.py output below the given folders (also when a copy from the cluster landed a folder deeper)
and prints, per run: the sanity check, top-1 and pairwise win rate per condition, scorer and kind, the mutation
operators for the CLM head, and CLM's breakdown by action category.

    python precheck/analyze.py                  # all runs under exps/
    python precheck/analyze.py exps/precheck/v1.json
"""
import glob, json, os, sys


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


def report(f, d):
    cfg, res = d["config"], d["results"]
    print(f"\n##### {f}  (tag {d['tag']}, {d['counts']['items']} steps / {d['counts']['instances']} instances, k={cfg['k']}, "
          f"{cfg.get('gpu', 'no GPU')}, CLM {str(cfg.get('clm_commit', '?'))[:7]}{', DRY RUN' if cfg.get('dry_run') else ''})")
    print(f"full candidate sets: {d['counts']['full_sets']}   truncated states: {d['counts'].get('truncated_share')}")
    conds = sorted({(r["max_len"], r["format"]) for r in res if r["max_len"]}, key=lambda c: (-c[0], c[1] != "turn"))   # primary first
    print(f"\ntop-1 among 1+{cfg['k']} (chance {res[0]['chance']:.3f}) | pairwise win rate, 95 % interval over instances")
    print(f"{'scorer':8s} {'kind':10s} " + " ".join(f"{f'{L}/{fm}':>33s}" for L, fm in conds))
    for sc in ("clm", "raw", "lexical"):
        for kd in ("random", "neighbour", "mutation"):
            cells = []
            for L, fm in conds:
                r = next((r for r in res if r["scorer"] == sc and r["kind"] == kd and r["format"] == fm and r["max_len"] in (L, 0)), None)
                cells.append(f"{ci(r['top1'], r['top1_ci']):>16s} | {r['pairwise']:.3f}" if r else " " * 33)
            print(f"{sc:8s} {kd:10s} " + " ".join(f"{c:>33s}" for c in cells))
    L, fm = conds[0]
    for kd in ("mutation", "neighbour"):
        r = next((r for r in res if (r["max_len"], r["format"], r["scorer"], r["kind"]) == (L, fm, "clm", kd)), None)
        if not r: continue
        print(f"\nclm {kd}, {L}/{fm}: pairwise win rate per {'operator' if kd == 'mutation' else 'position'}")
        for op, v in sorted(r["by_op"].items(), key=lambda kv: kv[1]["pairwise"]):
            print(f"  {op:24s} n={v['n']:5d}  {ci(v['pairwise'], v['ci'])}")
        print(f"clm {kd}, {L}/{fm}: by action (pairwise / top-1, n)")
        for sub, v in r["by_sub"]["pairwise"].items():
            t = r["by_sub"]["top1"].get(sub)
            print(f"  {sub:24s} {v['mean']:.3f} / {t['mean'] if t else float('nan'):.3f}   n={v['n']}")
        if r.get("ece") is not None: print(f"  ECE of the top probability (full sets): {r['ece']:.3f}, mean p(true) {r['p_true']:.3f}")


if __name__ == "__main__":
    runs = load(sys.argv[1:])
    if not runs: sys.exit("no pre-check JSON found")
    for f, d in runs: report(f, d)
