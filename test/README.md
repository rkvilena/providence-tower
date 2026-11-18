# Test Suite Guide

This directory contains three distinct layers of testing and evaluation:

1. **Fast component checks** use pytest and should not make live model/provider
   calls. Run these during normal development.
2. **The golden RAG benchmark** sends reviewed questions through the configured
   end-to-end RAG service. It is slower and may require credentials, external
   services, and paid model calls.
3. **Optional manual/deeper evaluation** covers a specific compatibility
   question today. LLM-as-judge and Ragas are not part of this suite.

Keeping these layers separate helps distinguish a broken component from a
change in end-to-end retrieval or answers. Retrieval metrics, reviewed answer
checks, latency, and failures are reported as separate signals; none alone is
a complete answer-quality verdict.

## Layer 1: Fast component checks

Run from the project root:

```text
python -m pytest test/unit -q
```

These checks are intended to be quick and deterministic. They use fakes where
external services would otherwise be needed.

| File | What it checks |
| --- | --- |
| `test/unit/test_reranker.py` | Rerank requests, score-to-hit mapping, fallback order, fetcher rerank/merge behavior, and preservation of pre-rerank candidates by query. |
| `test/unit/test_planner.py` | Planner helpers, modes, entity extraction, deduplication, and planner settings restore. |
| `test/unit/test_fetcher_cloudflare_embedding.py` | Query embedding through Cloudflare, including configuration and error paths. |
| `test/unit/test_cloudflare_workers_ai.py` | The shared Cloudflare Workers AI client: embeddings, reranking, errors, and secret handling. |
| `test/unit/test_import_boundaries.py` | Runtime imports stay free of the heavy local model stack. |
| `test/unit/test_benchmark_metrics.py` | Set-based retrieval metrics, MRR, Recall@5, duplicate IDs, and summary aggregation. |
| `test/unit/test_benchmark_evaluator.py` | Reviewed `any_of`/`all_of` answer checks, item-bound question 16 phrases, and unevaluated cases. |
| `test/unit/test_golden_dataset.py` | JSON/JSONL round-trip content and order, deterministic generation, and stale/missing `--check` reporting. |
| `test/unit/test_benchmark_report.py` | Markdown report contents, metadata, and missing-input handling. |

Passing these tests does not measure end-to-end answer quality on the golden
questions.

## Layer 2: Golden RAG benchmark

The editable dataset source is `test/GOLDEN-DATASET.json`, a readable JSON array
of reviewed questions. `test/GOLDEN-DATASET.jsonl` is generated from it and
checked in; the runner and evaluator read the JSONL, so their paths and behavior
are unchanged. Each record has a question, expected context chunk ID(s), a
reference answer, and reviewed `answer_checks` for explicit facts or values that
are suitable for deterministic checking.

Edit only the JSON source, then regenerate the JSONL from the project root:

```text
python test/sync_golden_dataset.py
```

To verify the checked-in JSONL matches the source without rewriting it:

```text
python test/sync_golden_dataset.py --check
```

`--check` exits unsuccessfully when the JSONL is stale or missing, making it
usable in a pre-run or CI check. Regenerate and commit the JSONL whenever the
JSON source changes; do not hand-edit the JSONL.

### One-command benchmark and evaluation

From the project root, validate the generated dataset, run the benchmark, and
evaluate its saved answers with one command:

```text
python -m test.run_benchmark
```

The workflow runs `python test/sync_golden_dataset.py --check` first. If the
generated JSONL is stale or missing, it stops without rewriting the file; run
`python test/sync_golden_dataset.py` explicitly to regenerate it before trying
again. Once the check passes, the workflow runs the benchmark and then the
deterministic answer-check evaluator.

An optional `--run-id <id>` labels the run. If omitted, the runner supplies its
timestamp-based run ID. The workflow forwards the runner options `--ids`,
`--limit`, `--warmup`, and `--output-dir`; `--ids` and `--limit` are mutually
exclusive. For example:

```text
python -m test.run_benchmark --run-id smoke --ids 1,10,12 --warmup
python -m test.run_benchmark --limit 5 --output-dir test/results/first-five
```

The evaluator writes `quality-results.jsonl` and `quality-summary.json`, then
refreshes the final `report.md`. By default, the report is at
`test/results/<run_id>/report.md`; with `--output-dir`, it is in that selected
run directory. The same directory also contains the runner's `raw-results.jsonl`
and `summary.json`.

### Standalone lower-level commands

The existing commands remain available when you want to run a stage separately.
For example, run the benchmark directly from the project root:

```text
python -m test.benchmark_runner --run-id baseline
```

The runner's useful options are:

- `--ids 1,10,12` runs selected questions.
- `--limit 5` runs the first five questions.
- `--warmup` sends one discarded query before measuring.
- `--output-dir <path>` changes where results are written.

`--ids` and `--limit` are mutually exclusive. By default, a run writes to
`test/results/<run_id>/`:

| Output | Purpose |
| --- | --- |
| `raw-results.jsonl` | Per-question answers, expected/retrieved/used chunk IDs, pre-rerank candidate IDs grouped by search query, timings, retrieval metrics, status, and errors. |
| `summary.json` | Run counts, latency and retrieval aggregates, token/configuration details, dataset hash, and available reproducibility metadata. |
| `report.md` | Readable run overview, latency and retrieval tables, question results, candidate pools, and failures. The runner creates this automatically. |

Candidate pools are recorded before reranking, grouped by search query. This
makes it possible to inspect whether runs offered the reranker the same
candidates; it does not itself guarantee that a comparison is like-for-like.
The benchmark reports precision/recall against expected chunk IDs and ranking
metrics (MRR and Recall@5) against the final ranked list. Retrieval metrics
describe retrieval, not answer correctness.

The runner records the code revision and tracked working-tree state, configured
embedding/query-embedding/reranker/generation model identifiers, and the
dataset hash. Corpus and index versions are reported as unavailable when they
cannot be determined. These details help identify a run but do not guarantee
that every external dependency or corpus state can be reproduced.

## Reviewed answer checks

After a benchmark, run the deterministic evaluator:

```text
python -m test.benchmark_evaluator --run-dir test/results/baseline
```

It writes `quality-results.jsonl` and `quality-summary.json`, then refreshes
`report.md`. `any_of` checks pass when at least one reviewed alternative is
present; `all_of` checks require every listed fact/value. A failed run, empty
answer, or question without reviewed checks is marked unevaluated.

These checks cover only selected explicit facts and values. They are not a
complete or authoritative answer-quality verdict. Token overlap is retained
only as a diagnostic; it should not be read as proof that differently worded
answers are wrong or right.

You can also regenerate the Markdown report directly:

```text
python -m test.benchmark_report --run-dir test/results/baseline
```

### How to read `report.md`

Start with **Run overview** and **Reproducibility metadata**. Confirm the run
ID, question selection, dataset hash, model settings, and code revision are the
ones you expected. `unavailable` means the runner could not identify that
version; it does not mean two runs used the same corpus or index. The working
tree flag covers tracked changes only.

Then read the sections as separate kinds of evidence:

- **Latency** is elapsed time. Compare runs only under reasonably similar
  conditions; a slower answer is not by itself a retrieval or answer-quality
  regression.
- **Retrieval** is measured against the expected chunk IDs in the dataset.
  Precision is the fraction of retrieved unique chunks that are expected;
  recall is the fraction of expected chunks that were retrieved. Hit rate asks
  whether any expected chunk appeared, and full recall asks whether all did.
  These set-based measures ignore order.
- **MRR** rewards placing the first expected chunk near the top: rank 1 gives
  1.0, rank 2 gives 0.5, rank 3 gives about 0.33, and no expected chunk gives
  0. **Recall@5** is the fraction of expected chunks found among the first five
  unique retrieved IDs. These ranking metrics use the final ranked list, after
  reranking when enabled. Duplicate IDs count once for ranking metrics.
- **Answer checks** say whether the answer contains selected, reviewed facts
  or values. A pass means those checks matched—not that the whole answer is
  correct, complete, or well-supported. A fail means one or more checks did
  not match; it can indicate an answer problem, or a check that needs review
  (for example, an accepted alias or wording was not included). `Unevaluated`
  means there is no verdict from these checks for that question.
- **Token overlap** is a rough text-similarity diagnostic only. Low overlap
  can be normal for a good paraphrase, and high overlap does not establish
  correctness.
- **Pre-rerank candidate pool** shows the chunk IDs retrieved for each search
  query before reranking. If two runs have different pools, a change in final
  ranking is not an isolated reranker comparison: the reranker received
  different candidates. If the pools match but the final order changes, the
  reranking or subsequent merge/order behavior is a more relevant place to
  investigate.
- **Run failures** mean the question did not complete successfully; inspect
  its error before interpreting its retrieval or answer fields.

For a surprising result, a practical order is:

1. Check the run overview, selected question IDs, dataset hash, model settings,
   and metadata for an unintended configuration difference.
2. Open that question's details. Compare expected chunks, the pre-rerank
   candidate pool, final retrieved chunks, and the individual answer checks.
3. Look at the answer itself and the recorded error (if any). Check whether a
   failed phrase check is a genuine missing fact or just an unreviewed
   equivalent wording.
4. For a before/after comparison, confirm both runs cover the same questions
   and inspect candidate pools. Also confirm corpus/index versions separately
   when the report says they are unavailable.
5. If the summary looks inconsistent with a question detail, inspect
   `raw-results.jsonl` and `quality-results.jsonl`; the Markdown report is a
   readable view of those saved data, not another evaluator.

## Comparing runs

Compare two saved benchmark runs:

```text
python -m test.benchmark_compare --baseline test/results/baseline --candidate test/results/candidate
```

Add `--per-question` to show per-question latency and recall changes. Inspect
each run's `report.md` and pre-rerank candidate pools when comparing rerankers.
A comparison is most informative when the runs use the same dataset, the same
questions, and comparable corpus/index and model configurations. The comparison
command does not automatically validate those conditions or compare candidate
pools for you.

## Layer 3: Optional manual check and future evaluation

`test/manual_bge_compatibility.py` is an optional compatibility smoke check for
the local and Cloudflare BGE embedding spaces. It needs local embedding
dependencies and Cloudflare credentials:

```text
python -m test.manual_bge_compatibility
```

This checks embedding-space compatibility, not model quality. There is
currently no LLM-as-judge or Ragas evaluation. Either could be considered later
as a separate optional signal; neither replaces the reviewed dataset, existing
benchmark runner, deterministic checks, or pytest suite.

## Remaining limitations

- The benchmark stores chunk IDs, not the retrieved chunk text. Context-grounded
  judgment therefore needs the exact corpus/index or another reliable way to
  reconstruct the text.
- Corpus and index versions may be unavailable, so matching model and dataset
  metadata alone may not be enough to reproduce a run exactly.
- Reviewed answer checks are intentionally narrow. Questions or answer aspects
  they do not cover remain unevaluated by those checks.
- The comparison command does not automatically verify run compatibility or
  report candidate-pool differences; use the saved metadata and per-query pools
  when assessing reranker comparisons.
