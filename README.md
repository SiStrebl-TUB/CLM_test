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

**Metrics.** Top-1 among 1 + 10 (full sets; chance 1/11 — the same form as CLM's 69.2 % on held-out
questions with 10 hard negatives), pairwise win rate against each negative (all steps), MRR, p(true), ECE of
the top probability; per mutation operator and per action type; 95 % intervals from a bootstrap over
instances.

Tokens, embeddings and heads come from CLM's own code at a pinned commit (`Recipe`, `OfflineBackend`,
`HeadPair`); `run.py` checks on the first steps that its cut equals `Recipe(max_len)`.

### Reading the result — proposed before the run

Primary cell: CLM head, budget 8192, format `turn`, `mutation`.

1. **Sanity.** Top-1 on `random` ≥ 0.8. Otherwise suspect the rendering of states or actions, not the model.
2. **Headroom.** Top-1 on `mutation` ≤ 0.6: clear headroom, go on to the A/B training experiment. ≥ 0.9:
   little headroom, hard negatives for actions are not the lever. In between: decide on the edit operators
   (`edit.{old,new,file}.*`), the cleanest negatives.
3. **Heads.** CLM should beat `raw` and `lexical` on `mutation`; if not, the heads add nothing to this
   discrimination.

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
bash cluster_setup.sh            # once, on a login node
sbatch cluster_precheck.sh v1    # ~1-2 h on one GPU with >= 24 GB
python precheck/analyze.py       # after copying exps/precheck/v1.json back
```

## Layout

```
precheck/data.py      trajectories -> steps, states, action rendering, natural negatives
precheck/mutate.py    near-miss operators
precheck/run.py       candidate sets -> CLM recipe -> embeddings -> scores -> exps/precheck/<tag>.*
precheck/analyze.py   tables from the JSONs
tests/                unit tests (no GPU, no downloads)
cluster_setup.sh      one-time environment, weights, data
cluster_precheck.sh   the SLURM job
```

Pinned: CLM `bb42c6c` (2026-09-24), head `CLM_v0.1-8B.pt` (sha256 `b2b4a8c9…`), `Qwen/Qwen3-8B`.
