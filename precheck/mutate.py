"""Near misses of an agent action: small edits of the true tool call that change what it does.

The operators edit the call's arguments with material from the state -- paths and identifiers the agent has
already seen -- so a negative looks as plausible as the true call and an unknown file name does not give it
away. mutations() pools the variants, drops any that equal the true call after whitespace normalisation, and
takes k round-robin over operators, so no single kind fills a candidate set when others exist. Paths get up
to k alternatives: for a simple call (view a file, run a script) another file is the natural near miss.

  bash.path     another path from the state in place of one in the command (other file, other directory)
  bash.flag     one flag removed (sed -i -> sed, grep -rn -> grep; some are cosmetic, e.g. pytest -v)
  bash.num      one number changed (line ranges, head -n)
  bash.pattern  one identifier inside a quoted string replaced (grep / sed patterns, -k selectors)
  bash.chain    the last segment of a && / ; / | chain dropped
  bash.code.*   edits of inline code (python -c "..."), as for the edit texts below
  edit.path     another path from the state (another file to view or edit)
  edit.range    the view window moved by its own width
  edit.line     the insert position moved by 5 lines
  edit.{old,new,file}.{flip,ident,num,dropline}   the text an edit matches / writes / creates: an operator
                flipped (== / !=, and / or, True / False, is None / is not None, ...), an identifier swapped,
                a number changed, a line dropped

Not every variant is wrong in effect: a flag or a number in an exploratory command can leave a near-equivalent
call, and viewing another file can be a reasonable step too. run.py therefore reports the win rate per
operator; the edits of written code are the clean cases.
"""
import random, re
from precheck.data import canon

KEYWORDS = {"self", "None", "True", "False", "return", "import", "from", "class", "def", "print", "else", "elif", "while", "with",
            "pass", "break", "continue", "lambda", "yield", "assert", "raise", "except", "finally", "global", "nonlocal", "async",
            "await", "workspace", "python", "python3", "pytest", "grep", "find", "head", "tail", "echo", "true", "false", "null", "none"}
IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{3,}\b")
NUM_RE = re.compile(r"(?<![\w.])\d+(?!\.?\d)")
FLAG_RE = re.compile(r"(?<=\s)--?[A-Za-z][\w-]*(?:=(?:\"[^\"]*\"|'[^']*'|\S+))?")
PATHTOK_RE = re.compile(r"[\w./@+~-]+")
EXT_RE = re.compile(r"\.(py|pyi|pyx|txt|md|rst|cfg|toml|ini|json|ya?ml|sh|c|h|cc|cpp|hpp|js|ts|go|rs|java|rb|html|css|csv|in|lock)$")
QUOTED_RE = re.compile(r"'([^'\n]{1,200})'|\"([^\"\n]{1,200})\"")
CHAIN_RE = re.compile(r"\s(&&|\|\||;|\|)\s")
FLIPS = [(re.compile(p), r) for p, r in [(r"==", "!="), (r"!=", "=="), (r">=", "<"), (r"<=", ">"), (r"(?<=\s)<(?=\s)", ">="),
                                         (r"(?<=\s)>(?=\s)", "<="), (r"\band\b", "or"), (r"\bor\b", "and"), (r"\bTrue\b", "False"),
                                         (r"\bFalse\b", "True"), (r"\bis not None\b", "is None"), (r"\bis None\b", "is not None"),
                                         (r"\bnot (?!None\b|in\b)", "")]]


# ----------------------------------------------------------------------------- material from the state
def is_path(t):
    t = t.rstrip(".,:;")
    if len(t) < 3 or t[0] == "-" or "//" in t or set(t) <= set("./~"): return False
    return bool(EXT_RE.search(t)) or ("/" in t and re.search(r"[A-Za-z]{2}", t) is not None)


def _recent(xs):
    seen, out = set(), []
    for x in reversed(xs):
        if x not in seen: seen.add(x); out.append(x)
    return out


def context(blob, n_paths=300, n_idents=500):
    """paths and identifiers seen in a state, most recent first"""
    paths = _recent([t.rstrip(".,:;") for t in PATHTOK_RE.findall(blob) if is_path(t)])
    idents = _recent([w for w in IDENT_RE.findall(blob) if w not in KEYWORDS])
    return {"paths": paths[:n_paths], "idents": idents[:n_idents]}


def _ext(p):
    m = EXT_RE.search(p)
    return m.group(1) if m else ""


def _alt_paths(p, pool, rng, n):
    """up to n other paths of the state with the same extension and the same absolute / relative form"""
    same = [q for q in pool if q != p and q.startswith("/") == p.startswith("/") and _ext(q) == _ext(p)]
    return rng.sample(same, min(n, len(same)))


def _sub(s, m, new):
    return s[:m.start()] + new + s[m.end():]


def _shift(s, rng):
    v = int(s)
    opts = [x for x in (v + 1, v - 1, v + 10, 2 * v, v // 2) if x >= 0 and x != v]
    return str(rng.choice(opts))


# ----------------------------------------------------------------------------- operators
def code_edits(text, ctx, rng, n=3):
    """(kind, new text): an operator flipped, an identifier swapped for one of the state, a number changed, a line dropped"""
    out = []
    hits = [(m, rep) for pat, rep in FLIPS for m in pat.finditer(text)]
    out += [("flip", _sub(text, m, rep)) for m, rep in rng.sample(hits, min(n, len(hits)))]
    pool, ids = ctx["idents"][:200], [m for m in IDENT_RE.finditer(text) if m.group() not in KEYWORDS]
    for m in rng.sample(ids, min(n, len(ids))):
        alts = [w for w in pool if w != m.group()]
        if alts: out.append(("ident", _sub(text, m, rng.choice(alts))))
    nums = list(NUM_RE.finditer(text))
    out += [("num", _sub(text, m, _shift(m.group(), rng))) for m in rng.sample(nums, min(n, len(nums)))]
    lines = text.split("\n"); full = [i for i, l in enumerate(lines) if l.strip()]
    if len(full) >= 2: out += [("dropline", "\n".join(lines[:i] + lines[i + 1:])) for i in rng.sample(full, min(n, len(full)))]
    return out


def bash_edits(cmd, ctx, rng, n=3, n_paths=10):
    out = []
    toks = [m for m in PATHTOK_RE.finditer(cmd) if is_path(m.group())]
    for m in toks:
        tok = m.group().rstrip(".,:;"); end = m.start() + len(tok)
        out += [("path", cmd[:m.start()] + q + cmd[end:]) for q in _alt_paths(tok, ctx["paths"], rng, n_paths)]
    for m in FLAG_RE.finditer(cmd):
        left, right = cmd[:m.start()].rstrip(" "), cmd[m.end():]
        out.append(("flag", left + ("" if not right or right[0] == " " else " ") + right))
    spans = [(m.start(), m.end()) for m in toks]
    nums = [m for m in NUM_RE.finditer(cmd) if not any(a <= m.start() < b for a, b in spans)]
    out += [("num", _sub(cmd, m, _shift(m.group(), rng))) for m in rng.sample(nums, min(n, len(nums)))]
    pool = ctx["idents"][:200]
    quoted = list(QUOTED_RE.finditer(cmd))
    for m in rng.sample(quoted, min(n, len(quoted))):
        g = 1 if m.group(1) is not None else 2
        ids = [x for x in IDENT_RE.finditer(m.group(g)) if x.group() not in KEYWORDS]
        if not ids: continue
        x, off = rng.choice(ids), m.start(g)
        alts = [w for w in pool if w != x.group()]
        if alts: out.append(("pattern", cmd[:off + x.start()] + rng.choice(alts) + cmd[off + x.end():]))
    seps = list(CHAIN_RE.finditer(cmd))
    if seps: out.append(("chain", cmd[:seps[-1].start()].rstrip()))
    if "\n" in cmd: out += [("code." + k, t) for k, t in code_edits(cmd, ctx, rng, n)]
    return out


def editor_edits(args, ctx, rng, n=3, n_paths=10):
    out, cmd = [], args.get("command")
    out += [("path", {**args, "path": q}) for q in _alt_paths(args["path"], ctx["paths"], rng, n_paths)]
    vr = args.get("view_range")
    if cmd == "view" and isinstance(vr, list) and len(vr) == 2 and all(isinstance(v, int) for v in vr):
        a, b = vr; w = 50 if b == -1 else max(10, b - a + 1)
        out += [("range", {**args, "view_range": [a + d, -1 if b == -1 else b + d]}) for d in (w, -w) if a + d >= 1]
    if cmd == "insert" and isinstance(args.get("insert_line"), int):
        il = args["insert_line"]
        out += [("line", {**args, "insert_line": v}) for v in sorted({il + 5, max(0, il - 5)} - {il})]
    for field, tag in (("old_str", "old"), ("new_str", "new"), ("file_text", "file")):
        if cmd != "view" and isinstance(args.get(field), str) and args[field].strip():
            out += [(f"{tag}.{k}", {**args, field: t}) for k, t in code_edits(args[field], ctx, rng, n)]
    return out


def mutations(a, ctx, k=10, seed=0):
    """[(operator, action)]: up to k distinct near misses of action a, round-robin over operators"""
    rng, bash = random.Random(seed), a["name"] == "execute_bash"
    raw = bash_edits(a["args"]["command"], ctx, rng, n_paths=k) if bash else editor_edits(a["args"], ctx, rng, n_paths=k)
    seen, by_op = {canon(a)}, {}
    for label, new in raw:
        m = {**a, "args": ({**a["args"], "command": new} if bash else new)}
        c = canon(m)
        if c not in seen: seen.add(c); by_op.setdefault(("bash." if bash else "edit.") + label, []).append(m)
    ops = sorted(by_op); rng.shuffle(ops); out = []
    while len(out) < k and any(by_op.values()):
        for o in ops:
            if by_op[o] and len(out) < k: out.append((o, by_op[o].pop(0)))
    return out
