# Scientific-agent evaluation environment

A tiny deterministic sandbox for testing whether a scientific agent follows an analysis protocol rather than merely producing a plausible final answer.

## Task

The agent receives a small RNA-seq-like count table and must:

1. inspect the experiment
2. normalize counts with log1p CPM
3. run Welch's t-test
4. apply Benjamini-Hochberg multiple-testing correction
5. submit the significant genes

The environment is stateful, replayable, and deterministically graded.

## Run

```bash
pip install -r requirements.txt
python run_trace.py traces/good_trace.json
python run_trace.py traces/bad_no_correction.json
python run_trace.py traces/bad_raw_counts.json
```

The good trace scores **1.0**. The two bad traces terminate in reproducible errors.

## Why this pattern matters

A scientific agent can produce a plausible final answer while silently violating the analysis protocol. An addressable sandbox captures:

- exact input state
- exact tool order
- exact parameters
- deterministic state transitions
- deterministic grading
- replayable traces

A natural next step is to generate perturbations automatically: missing normalization, swapped labels, incorrect multiple-testing correction, train/test leakage, wrong reference build, and similar failures.

## Files

- `science_env.py`: stateful sandbox and grader
- `run_trace.py`: trace replay runner
- `traces/good_trace.json`
- `traces/bad_no_correction.json`
- `traces/bad_raw_counts.json`

## Limitation

This is deliberately small and pedagogical. It demonstrates the environment/evaluation pattern, not a production RNA-seq pipeline.
