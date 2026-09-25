# CLMHardNegatives

Would hard negatives for *agent actions* help Contrastive Language Models
([CLM](https://github.com/Contrastive-LM/CLM), Kwok et al. 2026)? CLM uses hard negatives only in
mid-training, on question–answer pairs; its agentic post-training contrasts each step with the other steps
in the batch. Before training anything, the pre-check asks whether there is headroom at all.

## Pre-check

**Question.** How well does the released CLM-8B head separate the action an agent took from near misses,
on agent steps it was not trained on?

**Data.** [nebius/SWE-rebench-openhands-trajectories](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories):
67,074 trajectories of Qwen3-Coder-480B in OpenHands, binary `resolved`, cc-by-4.0. It is not among the ADP
sources of CLM's post-training (ADP does contain `nebius/SWE-agent-trajectories`, a different set). One
resolved trajectory per instance, two random steps each, 2000 steps. A step is an assistant message with
one call to `execute_bash` or `str_replace_editor`; its state is every message before it.

**Candidate sets.** The taken action plus k = 10 negatives, one set per kind:

| kind | negatives | role |
|---|---|---|
| `random` | true actions of the same tool from other tasks | sanity check: must be easy |
| `neighbour` | other actions of the same trajectory, nearest first | what CLM's post-training separates |
| `mutation` | near misses of the true call ([precheck/mutate.py](precheck/mutate.py)) | what hard negatives would add |

Mutations edit the call with material from the state (its paths, its identifiers), so an unknown name does
not give them away: another file, a dropped flag, a changed number or pattern, a truncated chain, and for
edits a flipped operator, swapped identifier, changed number or dropped line in the text that is matched or
written.

**Conditions.** Two unknowns of CLM's recipe are crossed: the state budget (2048 tokens as served, 8192 as
in fine-tuning; states are cut from the front) and the action format (`turn` = thought + `<tool_call>`
block as the Qwen3 chat template writes it, `call` = the block alone).

**Scorers.** The CLM head; the raw Qwen3 embeddings (CLM's `clm-raw` ablation); a lexical baseline (share of
the action's tokens that occur in the state's last 8000 characters).

**Metrics.** Pairwise win rate of the true action against each negative (pooled over all negatives of all
steps), top-1 among 1 + 10 (full sets; chance 1/11 — the form of CLM's 69.2 % on held-out questions with 10
hard negatives), MRR, p(true), ECE of the top probability; per mutation operator and per action type; 95 %
intervals from a bootstrap over instances.

Tokens, embeddings and heads come from CLM's own code at a pinned commit (`Recipe`, `OfflineBackend`,
`HeadPair`); `run.py` checks on the first steps that its cut equals `Recipe(max_len)`.

### Reading the result — fixed before the run (2026-09-25)

Pairwise win rates are pooled over all negatives; intervals are 95 % from a bootstrap over instances.
`python precheck/analyze.py` computes every step and prints the verdict.

0. **Cell.** The budget × format with the highest CLM pairwise win rate on `neighbour` sets: the kind closest
   to CLM's post-training, and independent of the mutations. Everything below is read in this cell.
1. **Sanity.** CLM top-1 on `random` ≥ 0.8. Otherwise the rendering of states or actions is off: stop there.
2. **Main number.** CLM pairwise win rate on `mutation`, next to `raw` and `lexical` (if the head does not
   beat them, it adds nothing to this discrimination).
3. **Ceiling.** The 60 sets in `<tag>_examples.md` are labelled in `<tag>_audit.csv` (Claude all, Simon 15
   as a cross-check; agreement and kappa are printed): is a negative as good as the true action? With q the
   share labelled *same* (unclear = 1/2), a perfect scorer reaches 1 − q/2. Operators with q > 0.3 on at
   least 10 labels are excluded and fixed before the A/B test.
4. **Verdict.** Headroom = ceiling − CLM (without flagged operators). Above 5 pp: GO, run the A/B training
   experiment. Below: STOP, hard negatives for actions are not the lever. If the interval of the headroom
   contains 5 pp: GREY ZONE, add the reference scorer (Qwen3-8B likelihood p(action | state)) first.

Why not top-1: if a share q of the negatives is as good as the true action, a perfect scorer reaches
(1 − (1 − q)^11) / (11q) top-1 among 1 + 10 — 0.62 at q = 10 % — but 1 − q/2 = 0.95 pairwise.

### Known limits

- Recipe unknowns (action format and state budget of the post-training) are crossed, not resolved.
- States are long: in a 30-trajectory sample all were longer than 2048 tokens and 95 % longer than 8192,
  so the head usually sees neither the system prompt nor the issue — as in CLM's recipe.
- Mutations are not verified by execution. Some are as good as the true call (another file to view, a
  cosmetic flag). Hence the per-operator rates and `<tag>_examples.md` for a manual check.
- `neighbour` measures imitation (which step the agent took), not correctness; `resolved` says the
  trajectory ended well, not that every step was good.

## Run

Locally, no GPU:

```bash
python -m unittest discover tests
curl -s "https://datasets-server.huggingface.co/rows?dataset=nebius/SWE-rebench-openhands-trajectories&config=default&split=train&offset=0&length=20" > /tmp/rows.json
python precheck/run.py --rows-json /tmp/rows.json --all-trajectories --dry-run --n-states 40 --tag smoke --out-dir /tmp/precheck
```

`--fake-encoder` instead of `--dry-run` uses the real tokenizer and head with a random encoder (needs
`external/CLM`, `transformers`, `torch`): everything but vLLM.

Cluster:

```bash
git clone https://github.com/SiStrebl-TUB/CLM_test.git /work/strebl/CLM_test   # the path the scripts expect
bash cluster_setup.sh            # once, on a login node
sbatch cluster_precheck.sh v1    # ~1-2 h on one GPU with >= 24 GB
```

Back home, with `v1.json`, `v1_items.jsonl.gz`, `v1_examples.md` and `v1_audit.csv` copied to
`exps/precheck/`: label `v1_audit.csv` (columns `label_a` / `label_b`: same, worse, unclear — reading the sets
in `v1_examples.md`), then `python precheck/analyze.py`.

## Layout

```
precheck/data.py      trajectories -> steps, states, action rendering, natural negatives
precheck/mutate.py    near-miss operators
precheck/run.py       candidate sets -> CLM recipe -> embeddings -> scores -> exps/precheck/<tag>.*
precheck/audit.py     the sheet for labelling the negatives by hand
precheck/analyze.py   tables and the decision
tests/                unit tests (no GPU, no downloads)
cluster_setup.sh      one-time environment, weights, data
cluster_precheck.sh   the SLURM job
```

Pinned: CLM `bb42c6c` (2026-09-24), head `CLM_v0.1-8B.pt` (sha256 `b2b4a8c9…`), `Qwen/Qwen3-8B`.
