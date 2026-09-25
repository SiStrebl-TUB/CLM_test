"""SWE-rebench OpenHands trajectories -> agent steps for the CLM pre-check.

Source: nebius/SWE-rebench-openhands-trajectories (67,074 trajectories of Qwen3-Coder-480B in OpenHands v0.54,
binary field `resolved`, cc-by-4.0, one 2 GB parquet file). It is not among the ADP sources, so the released
CLM head has most likely not been post-trained on it.

A step is an assistant message with exactly one tool call to execute_bash or str_replace_editor. Its state is
every message before it, as chat messages: the layout of CLM's agentic transitions (train/adapters.py), which
embed_utils.Recipe renders with the Qwen3 chat template and cuts from the front. Its action is that call as
text. How CLM's post-training rendered actions is not published, so there are two formats: "turn" is the
assistant turn as the Qwen3 chat template writes it (thought, then the <tool_call> block), "call" the block
alone. Every candidate, true or negative, goes through the same renderer, so formatting cannot give the true
action away.
"""
import json, os, random, re

DATASET, PARQUET = "nebius/SWE-rebench-openhands-trajectories", "trajectories.parquet"
TOOLS = ("execute_bash", "str_replace_editor")
FORMATS = ("turn", "call")
EDIT_COMMANDS = ("create", "str_replace", "insert", "undo_edit")


# ----------------------------------------------------------------------------- loading
def download(dest="exps/precheck/data"):
    path = os.path.join(dest, PARQUET)
    if not os.path.exists(path):
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(DATASET, PARQUET, repo_type="dataset", local_dir=dest)
    return path


def pick_rows(instances, resolved, n_traj, seed=0, resolved_only=True):
    """indices of one random trajectory per instance, for n_traj random instances"""
    by_inst = {}
    for i, (k, r) in enumerate(zip(instances, resolved)):
        if r == 1 or not resolved_only: by_inst.setdefault(k, []).append(i)
    rng = random.Random(seed); keys = sorted(by_inst); rng.shuffle(keys)
    return sorted(rng.choice(by_inst[k]) for k in keys[:n_traj])


def read_parquet(path, n_traj, seed=0, resolved_only=True):
    """the picked rows; only the row groups that hold them are decompressed (the file is 16 GB unpacked)"""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    meta = pf.read(columns=["instance_id", "resolved"]).to_pydict()
    pick = pick_rows(meta["instance_id"], meta["resolved"], n_traj, seed, resolved_only)
    cols, out, lo = ["trajectory_id", "instance_id", "repo", "resolved", "trajectory"], [], 0
    for g in range(pf.num_row_groups):
        hi = lo + pf.metadata.row_group(g).num_rows
        want = [i - lo for i in pick if lo <= i < hi]
        if want: out += pf.read_row_group(g, columns=cols).take(want).to_pylist()
        lo = hi
    return out


def read_rows_json(path, n_traj, seed=0, resolved_only=True):
    """the same from a JSON list of rows (datasets-server /rows output), for local tests"""
    rows = json.load(open(path))
    rows = [r.get("row", r) for r in (rows["rows"] if isinstance(rows, dict) else rows)]
    pick = pick_rows([r["instance_id"] for r in rows], [r["resolved"] for r in rows], n_traj, seed, resolved_only)
    return [rows[i] for i in pick]


# ----------------------------------------------------------------------------- steps
def parse_action(m):
    """{"name", "args", "text"} of an assistant message with exactly one call to a scored tool, else None"""
    calls = m.get("tool_calls") or []
    if m.get("role") != "assistant" or len(calls) != 1: return None
    name = calls[0]["function"]["name"]
    if name not in TOOLS: return None
    try: args = json.loads(calls[0]["function"]["arguments"])
    except (TypeError, ValueError): return None
    key = "command" if name == "execute_bash" else "path"
    if not isinstance(args, dict) or not isinstance(args.get(key), str) or not args[key].strip(): return None
    return {"name": name, "args": args, "text": (m.get("content") or "").strip()}


def steps_of(traj):
    """[(message index, action)] of the scored steps that have a user turn before them"""
    first_user = next((i for i, m in enumerate(traj) if m["role"] == "user"), None)
    if first_user is None: return []
    return [(i, a) for i, m in enumerate(traj) if i > first_user and (a := parse_action(m)) is not None]


def state_messages(traj, i):
    """the messages before step i, reduced to what the Qwen3 chat template reads (content always a string)"""
    out = []
    for m in traj[:i]:
        d = {"role": m["role"], "content": m.get("content") or ""}
        if m.get("tool_calls"):
            d["tool_calls"] = [{"type": "function", "function": {"name": t["function"]["name"], "arguments": t["function"]["arguments"]}}
                               for t in m["tool_calls"]]
        out.append(d)
    return out


def state_blob(msgs, tail=40000):
    """plain text of a state (contents and call arguments), its last `tail` characters"""
    parts = []
    for m in msgs:
        parts.append(m["content"]); parts += [t["function"]["arguments"] for t in m.get("tool_calls", [])]
    return "\n".join(parts)[-tail:]


# ----------------------------------------------------------------------------- actions
def call_text(name, args):
    """the <tool_call> block exactly as the Qwen3 chat template writes an assistant tool call"""
    return '<tool_call>\n{"name": "' + name + '", "arguments": ' + json.dumps(args, ensure_ascii=False) + "}\n</tool_call>"


def render_action(a, fmt):
    c = call_text(a["name"], a["args"])
    return a["text"] + "\n" + c if fmt == "turn" and a["text"] else c


def canon(a):
    """identity of a call for de-duplication: tool and arguments, whitespace runs collapsed"""
    norm = lambda v: re.sub(r"\s+", " ", v).strip() if isinstance(v, str) else v
    return json.dumps([a["name"], {k: norm(v) for k, v in a["args"].items()}], sort_keys=True, ensure_ascii=False)


def category(a):
    """explore / edit / run / other, from the editor command or the first program after any leading cd"""
    if a["name"] == "str_replace_editor": return "edit" if a["args"].get("command") in EDIT_COMMANDS else "explore"
    cmd = re.sub(r"^\s*(cd\s+\S+\s*(&&|;)\s*)+", "", a["args"]["command"]).strip()
    first = cmd.split()[0] if cmd else ""
    if re.search(r"(^|\s)(sed\s+-i|rm|mv|cp|mkdir|touch|patch|git\s+(apply|checkout|stash|reset))\b", cmd): return "edit"
    if first in ("echo", "cat", "printf") and re.search(r"(^|[^2&>])>\s*[\w/.]", cmd): return "edit"
    if first in ("python", "python3", "pytest", "pip", "pip3", "conda", "make", "tox", "bash", "sh", "npm", "node", "yarn", "cargo", "go"): return "run"
    if first in ("grep", "egrep", "rg", "find", "ls", "cat", "head", "tail", "wc", "tree", "sed", "awk", "git", "pwd", "which", "echo", "diff", "file", "stat", "xargs"): return "explore"
    return "other"


# ----------------------------------------------------------------------------- natural negatives
def neighbours(steps, j, k):
    """up to k other actions of the same trajectory, nearest steps first, distinct from the true action and each
    other; labelled past/future and near (1-2 steps away) or far"""
    seen, out = {canon(steps[j][1])}, []
    for jj in sorted((jj for jj in range(len(steps)) if jj != j), key=lambda jj: (abs(jj - j), jj > j)):
        c = canon(steps[jj][1])
        if c in seen: continue
        seen.add(c)
        out.append((("past" if jj < j else "future") + ("-near" if abs(jj - j) <= 2 else "-far"), steps[jj][1]))
        if len(out) == k: break
    return out


def random_negatives(pool, a0, instance, k, rng):
    """k true actions of the same tool from other instances: other repositories, other tasks -- the sanity check"""
    cand = [a for inst, a in pool[a0["name"]] if inst != instance]
    seen, out = {canon(a0)}, []
    for a in rng.sample(cand, min(len(cand), 4 * k)):
        c = canon(a)
        if c not in seen: seen.add(c); out.append(("random", a))
        if len(out) == k: break
    return out
