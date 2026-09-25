"""Hand check of the mutation negatives: is a negative as good as the true action?

    python precheck/audit.py exps/precheck/v1_examples.md      # -> exps/precheck/v1_audit.csv (run.py writes it too)

For each set in <tag>_examples.md, label the candidates m1..m10 against the true action m0 in the column label_a
(all sets) and, as a cross-check by a second person, in label_b (some sets): same = at this point the call would
serve the agent as well as m0; worse = it would not; unclear. German works too (gleich / schlechter / unklar).
Judge from the task and the state as shown. analyze.py reads <tag>_audit.csv next to <tag>.json: the share q of
"same" (unclear = 1/2) gives the ceiling 1 - q/2 of the pairwise win rate.
"""
import csv, os, re, sys

LABELS = {"same": 1.0, "gleich": 1.0, "worse": 0.0, "schlechter": 0.0, "unclear": 0.5, "unklar": 0.5}
HEAD_RE = re.compile(r"^## (\S+:\d+)  \(\S+, \d+ state tokens\)\s*$")
CAND_RE = re.compile(r"^- m(\d+) `([\w.-]+)` [+-]\d+\.\d+\s*$")
COLUMNS = ("sid", "cand", "op", "label_a", "label_b", "note")


def parse_examples(path):
    """[{sid, cand, op}] of the mutation negatives in an examples file. Task and state text sit in ~~~~ fences and
    are skipped; the strict patterns also keep files written with ``` fences readable"""
    rows, sid, fenced = [], None, False
    with open(path) as f:
        for line in f:
            if line.rstrip("\n") in ("~~~~text", "~~~~"): fenced = line.startswith("~~~~text"); continue
            if fenced: continue
            if (m := HEAD_RE.match(line)): sid = m.group(1)
            elif (m := CAND_RE.match(line)) and sid and int(m.group(1)) > 0:
                rows.append({"sid": sid, "cand": f"m{m.group(1)}", "op": m.group(2)})
    return rows


def write_sheet(examples_path, force=False):
    """<tag>_audit.csv with one empty row per negative; an existing sheet is kept, labels are never overwritten"""
    out = examples_path.replace("_examples.md", "_audit.csv")
    if os.path.exists(out) and not force:
        print(f"kept {out}: it exists (labels are not overwritten)"); return out
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, COLUMNS); w.writeheader()
        for r in parse_examples(examples_path): w.writerow({**r, "label_a": "", "label_b": "", "note": ""})
    return out


def read_labels(path, column="label_a"):
    """{(sid, cand): (op, value)} of the labelled rows; value 1 = same, 0 = worse, 1/2 = unclear"""
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            v = LABELS.get((r.get(column) or "").strip().lower())
            if v is not None: out[(r["sid"], r["cand"])] = (r["op"], v)
    return out


if __name__ == "__main__":
    if len(sys.argv) < 2: sys.exit(__doc__)
    print("written", write_sheet(sys.argv[1], force="--force" in sys.argv))
