"""python -m unittest discover tests   (no GPU, no downloads)"""
import json, os, random, sys, unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from precheck.data import call_text, canon, category, neighbours, parse_action, render_action, state_messages, steps_of
from precheck.mutate import bash_edits, code_edits, context, editor_edits, mutations


def msg(role, content="", call=None):
    m = {"role": role, "content": content, "name": None, "tool_call_id": None, "tool_calls": None}
    if call: m["tool_calls"] = [{"function": {"name": call[0], "arguments": json.dumps(call[1])}, "id": "x", "type": "function"}]
    return m


TRAJ = [msg("system", "You are OpenHands agent."), msg("user", "Fix the bug in /workspace/repo/pkg/core.py"),
        msg("assistant", "Let me look.", ("str_replace_editor", {"command": "view", "path": "/workspace/repo"})),
        msg("tool", "/workspace/repo/pkg/core.py\n/workspace/repo/pkg/util.py\n/workspace/repo/tests/test_core.py"),
        msg("assistant", "", ("think", {"thought": "hmm"})), msg("tool", "Your thought has been logged."),
        msg("assistant", "Search.", ("execute_bash", {"command": "cd /workspace/repo && grep -rn \"parse_value\" pkg/core.py | head -20"})),
        msg("tool", "pkg/core.py:12: def parse_value(raw_input, strict=True):"),
        msg("assistant", "Fix it.", ("str_replace_editor", {"command": "str_replace", "path": "/workspace/repo/pkg/core.py",
                                                            "old_str": "if value == None:\n    return default", "new_str": "if value is None:\n    return default"})),
        msg("tool", "The file has been edited."),
        msg("assistant", "", ("execute_bash", {"command": "cd /workspace/repo && python -m pytest tests/test_core.py -v"}))]
CTX = context("\n".join(m["content"] or "" for m in TRAJ) + "\npkg/util.py tests/test_core.py parse_value raw_input strict default_value")


class TestData(unittest.TestCase):
    def test_steps_skip_other_tools(self):
        s = steps_of(TRAJ)
        self.assertEqual([i for i, _ in s], [2, 6, 8, 10])
        self.assertEqual(s[1][1]["name"], "execute_bash")

    def test_multi_call_and_bad_json_are_skipped(self):
        m = msg("assistant", "", ("execute_bash", {"command": "ls"})); m["tool_calls"] *= 2
        self.assertIsNone(parse_action(m))
        m = msg("assistant", "", ("execute_bash", {"command": "ls"})); m["tool_calls"][0]["function"]["arguments"] = "{not json"
        self.assertIsNone(parse_action(m))

    def test_state_is_the_prefix_with_string_contents(self):
        st = state_messages(TRAJ, 6)
        self.assertEqual(len(st), 6); self.assertTrue(all(isinstance(m["content"], str) for m in st))
        self.assertEqual(st[2]["tool_calls"][0]["function"]["name"], "str_replace_editor")
        self.assertNotIn("tool_calls", st[3])

    def test_render_matches_qwen3_template(self):
        a = {"name": "execute_bash", "args": {"command": "ls -la"}, "text": "Look around."}
        call = '<tool_call>\n{"name": "execute_bash", "arguments": {"command": "ls -la"}}\n</tool_call>'
        self.assertEqual(call_text(a["name"], a["args"]), call)
        self.assertEqual(render_action(a, "turn"), "Look around.\n" + call)
        self.assertEqual(render_action(a, "call"), call)
        self.assertEqual(render_action({**a, "text": ""}, "turn"), call)

    def test_canon_ignores_whitespace_only(self):
        a = {"name": "execute_bash", "args": {"command": "ls  -la"}, "text": "x"}
        self.assertEqual(canon(a), canon({**a, "args": {"command": "ls -la "}, "text": "other"}))
        self.assertNotEqual(canon(a), canon({**a, "args": {"command": "ls -l"}}))

    def test_category(self):
        c = lambda cmd: category({"name": "execute_bash", "args": {"command": cmd}})
        self.assertEqual(c("cd /w && grep -rn x ."), "explore"); self.assertEqual(c("cd /w && python -m pytest"), "run")
        self.assertEqual(c("sed -i 's/a/b/' f.py"), "edit"); self.assertEqual(c("echo hi > f.txt"), "edit")
        self.assertEqual(c("grep x f 2>/dev/null"), "explore")
        self.assertEqual(category({"name": "str_replace_editor", "args": {"command": "view", "path": "/w"}}), "explore")

    def test_neighbours_distinct_and_labelled(self):
        s = steps_of(TRAJ); nb = neighbours(s, 1, 10)
        self.assertEqual(len(nb), 3)
        self.assertEqual(len({canon(a) for _, a in nb} | {canon(s[1][1])}), 4)
        self.assertEqual([op for op, _ in nb], ["past-near", "future-near", "future-near"])


class TestMutate(unittest.TestCase):
    CMD = "cd /workspace/repo && grep -rn \"parse_value\" pkg/core.py | head -20"

    def test_bash_operators(self):
        ed = bash_edits(self.CMD, CTX, random.Random(0))
        kinds = {k for k, _ in ed}
        self.assertTrue({"path", "flag", "num", "pattern", "chain"} <= kinds, kinds)
        self.assertIn("cd /workspace/repo && grep -rn \"parse_value\" pkg/core.py", [t for k, t in ed if k == "chain"])
        self.assertIn("cd /workspace/repo && grep \"parse_value\" pkg/core.py | head -20", [t for k, t in ed if k == "flag"])
        paths = [t for k, t in ed if k == "path"]         # only relative .py files of the state can replace pkg/core.py
        self.assertTrue(paths and all("pkg/util.py" in t or "tests/test_core.py" in t for t in paths), paths)
        self.assertTrue(all("head -20" not in t for k, t in ed if k == "num"))
        self.assertTrue(all('"parse_value"' not in t for k, t in ed if k == "pattern"))

    def test_path_swap_keeps_extension(self):
        ed = editor_edits({"command": "view", "path": "/workspace/repo/pkg/core.py"}, CTX, random.Random(0))
        paths = [a["path"] for k, a in ed if k == "path"]
        self.assertTrue(paths); self.assertTrue(all(p.endswith(".py") and p != "/workspace/repo/pkg/core.py" for p in paths))

    def test_view_range_and_code_edits(self):
        ed = editor_edits({"command": "view", "path": "/workspace/repo/pkg/core.py", "view_range": [10, 30]}, CTX, random.Random(0))
        self.assertIn([31, 51], [a["view_range"] for k, a in ed if k == "range"])
        ed = code_edits("if value is None:\n    return default", CTX, random.Random(0))
        self.assertIn(("flip", "if value is not None:\n    return default"), ed)
        self.assertIn(("dropline", "    return default"), ed)

    def test_mutations_distinct_bounded_deterministic(self):
        a = parse_action(TRAJ[8])
        m1, m2 = mutations(a, CTX, 10, seed="s"), mutations(a, CTX, 10, seed="s")
        self.assertEqual([canon(x) for _, x in m1], [canon(x) for _, x in m2])
        self.assertLessEqual(len(m1), 10)
        cs = [canon(x) for _, x in m1]
        self.assertEqual(len(set(cs)), len(cs)); self.assertNotIn(canon(a), cs)
        self.assertGreaterEqual(len({op for op, _ in m1}), 3)
        self.assertTrue(all(x["text"] == a["text"] for _, x in m1))

    def test_no_mutation_material_gives_empty(self):
        a = {"name": "execute_bash", "args": {"command": "pwd"}, "text": ""}
        self.assertEqual(mutations(a, {"paths": [], "idents": []}, 10), [])


class TestMetrics(unittest.TestCase):
    def test_summarize_by_hand(self):
        import numpy as np
        from precheck.run import summarize

        class Args: k, n_boot = 2, 200
        items = [dict(instance=f"i{j}", category="edit", sub="editor.create", n_tokens=100,
                      cands={"mutation": [("true", None), ("edit.path", None), ("edit.range", None)]}) for j in range(4)]
        S = [np.array(v) for v in ([1.0, 0.5, 0.2], [0.1, 0.5, 0.2], [0.5, 0.5, 0.2], [0.3, 0.1, 0.9])]
        r = summarize(items, {(2048, "turn", "clm", "mutation"): S}, {"clm": 10.0}, Args)[0]
        self.assertAlmostEqual(r["top1"], (1 + 0 + 0.5 + 0) / 4)          # a tie at the top counts 1/2
        self.assertAlmostEqual(r["pairwise"], (2 + 0 + 1.5 + 1) / 8)      # per negative; a tied negative counts 1/2
        self.assertAlmostEqual(r["mrr"], (1 + 1 / 3 + 2 / 3 + 1 / 2) / 4)
        self.assertAlmostEqual(r["by_op"]["edit.path"]["pairwise"], 0.625)
        self.assertAlmostEqual(r["by_op"]["edit.range"]["pairwise"], 0.5)
        self.assertAlmostEqual(r["chance"], 1 / 3)
        self.assertTrue(r["pairwise_ci"][0] <= r["pairwise"] <= r["pairwise_ci"][1])


class TestDecision(unittest.TestCase):
    """the README rule end to end on made-up outputs: 40 steps, each true action against one other-file view
    (edit.path) and one flipped edit (edit.new.flip)"""

    def run_rule(self, flip_scores, path_label):
        import contextlib, csv, gzip, io, tempfile
        from precheck.analyze import decide
        from precheck.audit import COLUMNS
        with tempfile.TemporaryDirectory() as tmp:
            f = os.path.join(tmp, "t.json")
            res = [dict(max_len=L, format="turn", scorer="clm", kind="neighbour", pairwise=p) for L, p in ((8192, 0.8), (2048, 0.7))]
            res.append(dict(max_len=8192, format="turn", scorer="clm", kind="random", top1=0.95))
            with open(f, "w") as h: json.dump({"kind": "clm_precheck", "results": res}, h)
            with gzip.open(f[:-5] + "_items.jsonl.gz", "wt") as g, open(f[:-5] + "_audit.csv", "w", newline="") as a:
                w = csv.DictWriter(a, COLUMNS); w.writeheader()
                for i in range(40):
                    sid = f"t{i}:5"; s = [0.5, 0.6, flip_scores]    # the true action loses to the other file
                    sets = {"mutation": {"ops": ["true", "edit.path", "edit.new.flip"], "scores": {"8192|turn|clm": s, "8192|turn|raw": s, "0|turn|lexical": s}}}
                    g.write(json.dumps({"sid": sid, "instance": f"inst{i}", "sets": sets}) + "\n")
                    w.writerow({"sid": sid, "cand": "m1", "op": "edit.path", "label_a": path_label})
                    w.writerow({"sid": sid, "cand": "m2", "op": "edit.new.flip", "label_a": "worse", "label_b": "worse" if i < 15 else ""})
            out = io.StringIO()
            with open(f) as h, contextlib.redirect_stdout(out): verdict = decide(f, json.load(h))
            return verdict, out.getvalue()

    def test_go_when_clm_misses_clean_negatives(self):
        verdict, log = self.run_rule(flip_scores=0.9, path_label="worse")   # CLM loses every pair, all negatives worse
        self.assertEqual(verdict, "GO", log)

    def test_stop_when_only_equivalent_negatives_beat_clm(self):
        verdict, log = self.run_rule(flip_scores=0.1, path_label="same")    # CLM loses only to views as good as the true one
        self.assertEqual(verdict, "STOP", log)
        self.assertIn("edit.path: q 1.00", log)                            # flagged and excluded
        self.assertIn("agreement 1.00", log)

    def test_audit_sheet_from_examples(self):
        import tempfile
        from precheck.audit import parse_examples
        md = ("# x\n## abc:12  (editor.view, 900 state tokens)\n\nTask:\n~~~~text\n## fake:1  (a, 1 state tokens)\n- m1 `edit.path` +0.100\n~~~~\n"
              "**mutation**\n\n- m0 `true` +0.300\n  ```text\n  call\n  ```\n- m1 `edit.path` +0.200\n- m2 `edit.range` -0.050\n**neighbour**\n\n- n1 `past-near` +0.1\n")
        with tempfile.NamedTemporaryFile("w", suffix="_examples.md", delete=False) as t: t.write(md)
        rows = parse_examples(t.name); os.unlink(t.name)
        # the heading and the candidate line inside the task fence are text, not a set; m0 and neighbours are not labelled
        self.assertEqual([(r["sid"], r["cand"], r["op"]) for r in rows], [("abc:12", "m1", "edit.path"), ("abc:12", "m2", "edit.range")])


if __name__ == "__main__":
    unittest.main()
