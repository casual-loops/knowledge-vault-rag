# Retrieval Evaluation

This project includes a deterministic retrieval evaluation framework for measuring semantic, lexical, and hybrid search quality against a public-safe synthetic dataset.

The framework is intended to detect meaningful retrieval regressions during development. It is not a universal benchmark for retrieval quality or production performance.

## Evaluation dataset

The synthetic evaluation dataset is stored at:

```text
evaluation/retrieval_cases.json
```

Each case defines a stable case identifier, evaluation category, query, retrieval mode, optional metadata filters, expected relevant source chunks, and graded relevance judgments.

The current dataset covers exact keyword retrieval, semantic retrieval, ambiguous queries, metadata-filtered retrieval, hybrid retrieval, and explicit no-result cases.

All judged sources come from the sanitized sample vault under:

```text
examples/sample-vault/
```

Evaluation data must remain public-safe. It must not include private vault content, credentials, internal infrastructure details, or identifying production data.

## Relevance judgments

Relevant chunks are identified by the combination of `source_path` and `chunk_index`.

The dataset uses this relevance scale:

| Value | Meaning |
| --- | --- |
| `3` | Highly relevant |
| `2` | Relevant |
| `1` | Marginally relevant |
| `0` | Not relevant |

The current precision, recall, and reciprocal-rank calculations treat every judgment greater than `0` as relevant.

The graded values are retained so future evaluation metrics can distinguish between highly relevant and marginally relevant results.

## Retrieval metrics

### Precision at K

Precision at K measures how many of the first `K` retrieved chunks are relevant.

```text
precision@k = relevant retrieved chunks / retrieved chunks considered
```

Higher precision means fewer irrelevant chunks appear in the evaluated result set.

Example: if three of the first five results are relevant, `precision@5` is `0.6`.

### Recall at K

Recall at K measures how many of the known relevant chunks were found within the first `K` results.

```text
recall@k = relevant retrieved chunks / all known relevant chunks
```

Higher recall means retrieval found more of the material that was judged relevant.

Example: if two relevant chunks exist and both appear in the first five results, `recall@5` is `1.0`.

For an explicit no-result case, recall is considered successful when no results are returned.

### Reciprocal rank

Reciprocal rank measures how early the first relevant result appears.

```text
reciprocal rank = 1 / rank of first relevant result
```

Examples:

```text
Relevant result at rank 1 = 1.0
Relevant result at rank 2 = 0.5
Relevant result at rank 4 = 0.25
No relevant result = 0.0
```

Mean reciprocal rank, or MRR, is the average reciprocal rank across evaluation cases.

## Retrieval modes

The evaluation framework supports `semantic`, `lexical`, and `hybrid` retrieval.

Semantic retrieval uses vector similarity. Lexical retrieval uses PostgreSQL full-text search. Hybrid retrieval combines semantic and lexical rankings with Reciprocal Rank Fusion.

Evaluation summaries are grouped by retrieval mode so changes can be compared independently.

## Running evaluations locally

The evaluation runner is:

```text
scripts/evaluate_retrieval.py
```

Run it from the repository root:

```powershell
python scripts/evaluate_retrieval.py
```

The command loads the synthetic evaluation dataset, executes each query using its configured retrieval mode, records ranked source chunks, calculates per-query metrics, calculates aggregate metrics by retrieval mode, writes machine-readable JSON output, prints a concise human-readable summary, and checks configured regression thresholds.

The default machine-readable output is written to:

```text
evaluation/results.json
```

A summary contains the number of evaluated cases plus mean precision at K, mean recall at K, and MRR for each retrieval mode represented by the dataset.

The exact values depend on the indexed synthetic corpus and configured retrieval behavior.

## Machine-readable results

The evaluation runner records both metrics and ranked retrieval results.

Each evaluated case includes case metadata, calculated retrieval metrics, ranked `source_path`, ranked `chunk_index`, and rank position.

This makes it possible to trace a metric change back to the specific chunks that moved, disappeared, or entered a result set.

## Regression thresholds

Regression thresholds are stored at:

```text
evaluation/thresholds.json
```

Thresholds are explicit configuration rather than hard-coded application logic.

The current configuration supports minimum aggregate precision by retrieval mode, minimum aggregate recall by retrieval mode, minimum aggregate mean reciprocal rank by retrieval mode, minimum per-case recall, and minimum per-case reciprocal rank.

If a configured threshold is breached, the evaluation command exits with a nonzero status.

Failure output identifies whether the failure occurred at retrieval-mode or individual-case scope, the affected retrieval mode or case identifier, the metric that failed, the actual value, and the configured minimum.

## Updating thresholds

Threshold changes should be intentional.

Before changing a threshold:

1. Run the evaluation against the current known-good implementation.
2. Review the per-query rankings and aggregate metrics.
3. Determine whether the metric change represents an expected retrieval improvement, a dataset change, or a regression.
4. Update `evaluation/thresholds.json` only when the new baseline is understood.
5. Commit the threshold change together with the code or evaluation-data change that justifies it.

Thresholds should not be relaxed solely to make a failing evaluation pass.

## Determinism

The automated evaluation suite is designed to avoid paid or nondeterministic dependencies.

Unit tests use deterministic fixtures and local test doubles. Regression thresholds should remain tied to reproducible retrieval behavior and should not depend on changing hosted-model behavior, paid APIs, or uncontrolled external services.

The runtime evaluation command can exercise the configured retrieval stack against the indexed synthetic corpus. Automated unit coverage remains deterministic even when optional external providers are unavailable.

## Limitations

### Small corpus

The sample vault is intentionally small. Results from this benchmark should not be assumed to predict retrieval behavior across a large personal vault.

### Synthetic content

The notes and queries are deliberately sanitized and artificial. Real personal knowledge can contain more ambiguity, inconsistent terminology, cross-linking, duplication, and incomplete metadata.

### Limited relevance judgments

Expected relevant chunks are manually defined. Human relevance judgments are subjective, especially for ambiguous queries.

### Retrieval-focused metrics

Precision, recall, and reciprocal rank measure retrieval behavior. They do not measure generated answer quality, factual accuracy, citation correctness, or user satisfaction.

### Binary treatment of graded relevance

Although the dataset stores graded relevance values, the current metrics treat every relevance value greater than zero as relevant.

A future evaluation phase could add a graded metric such as normalized discounted cumulative gain.

### Production evaluation remains separate

Before production vault indexing, evaluation should be expanded using private test cases that remain outside the public repository.

Private evaluation data must follow the same privacy boundaries as the production vault itself.

## Purpose

The evaluation framework provides a stable engineering feedback loop for retrieval changes.

Its central question is simple: did a retrieval change improve quality, preserve it, or make it worse?

The framework provides measurable evidence for that question without exposing private knowledge or requiring external model services.