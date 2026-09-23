"""
Tiny deterministic scientific-agent sandbox.

The task: identify truly differential genes from a small RNA-seq-like count table.
The environment is intentionally stateful. Correct answers require the agent to:
1) inspect metadata,
2) normalize counts with log1p CPM,
3) run Welch's t-test with Benjamini-Hochberg correction,
4) submit genes at FDR <= 0.05.

The point is not biological realism. The point is reproducible, gradeable failure states.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

def bh_adjust(p):
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return out

def make_data(seed=7):
    rng = np.random.default_rng(seed)
    genes = [f"G{i:02d}" for i in range(1, 31)]
    samples = [f"C{i}" for i in range(1, 9)] + [f"T{i}" for i in range(1, 9)]
    condition = np.array(["control"]*8 + ["treated"]*8)
    libsize = rng.lognormal(mean=0, sigma=0.25, size=16)

    base = rng.lognormal(mean=5.0, sigma=0.45, size=len(genes))
    fold = np.ones(len(genes))
    truth = {"G04": 3.2, "G11": 0.28, "G19": 2.8, "G27": 0.32}
    for g, fc in truth.items():
        fold[genes.index(g)] = fc

    counts = np.zeros((len(genes), len(samples)), dtype=int)
    for gi, g in enumerate(genes):
        for si, cond in enumerate(condition):
            mu = base[gi] * libsize[si] * (fold[gi] if cond == "treated" else 1.0)
            lam = max(mu * rng.lognormal(0, 0.16), 1)
            counts[gi, si] = rng.poisson(lam)

    return (
        pd.DataFrame(counts, index=genes, columns=samples),
        pd.DataFrame({"sample": samples, "condition": condition}).set_index("sample"),
        set(truth)
    )

@dataclass
class ScienceSandbox:
    seed: int = 7
    counts: pd.DataFrame = field(init=False)
    metadata: pd.DataFrame = field(init=False)
    truth: set = field(init=False)
    inspected: bool = field(default=False, init=False)
    normalized: pd.DataFrame | None = field(default=None, init=False)
    results: pd.DataFrame | None = field(default=None, init=False)
    events: list = field(default_factory=list, init=False)

    def __post_init__(self):
        self.counts, self.metadata, self.truth = make_data(self.seed)

    def reset(self):
        self.__post_init__()
        self.inspected = False
        self.normalized = None
        self.results = None
        self.events = []
        return {"status": "reset"}

    def inspect_experiment(self):
        self.inspected = True
        self.events.append(("inspect_experiment", {}))
        return {
            "n_genes": int(self.counts.shape[0]),
            "n_samples": int(self.counts.shape[1]),
            "conditions": self.metadata["condition"].value_counts().to_dict(),
            "sample_order": list(self.counts.columns),
            "note": "Counts are raw integer counts; library sizes differ."
        }

    def normalize_counts(self, method: str):
        self.events.append(("normalize_counts", {"method": method}))
        if not self.inspected:
            return {"error": "Inspect metadata before transforming the data."}
        if method != "log1p_cpm":
            return {"error": "Unsupported or unsafe demo method. Use log1p_cpm."}

        lib = self.counts.sum(axis=0)
        cpm = self.counts.div(lib, axis=1) * 1_000_000
        self.normalized = np.log1p(cpm)
        return {"status": "ok", "method": method, "shape": list(self.normalized.shape)}

    def differential_expression(self, test: str, correction: str):
        self.events.append(("differential_expression", {"test": test, "correction": correction}))
        if self.normalized is None:
            return {"error": "Normalize counts before differential expression."}
        if test != "welch_t":
            return {"error": "This sandbox's validated test is welch_t."}
        if correction != "BH":
            return {"error": "Multiple-testing correction is required; use BH."}

        ctrl = self.metadata.index[self.metadata.condition == "control"]
        trt = self.metadata.index[self.metadata.condition == "treated"]
        rows = []
        for g in self.normalized.index:
            a = self.normalized.loc[g, ctrl].values
            b = self.normalized.loc[g, trt].values
            stat, p = ttest_ind(b, a, equal_var=False)
            lfc = float(b.mean() - a.mean())
            rows.append((g, lfc, float(p)))
        res = pd.DataFrame(rows, columns=["gene", "log_expr_diff", "p_value"])
        res["q_value"] = bh_adjust(res["p_value"].values)
        res = res.sort_values(["q_value", "p_value"]).reset_index(drop=True)
        self.results = res
        return {"status": "ok", "top": res.head(8).round(5).to_dict(orient="records")}

    def submit(self, genes: list[str]):
        self.events.append(("submit", {"genes": genes}))
        if self.results is None:
            return {"error": "Run differential expression before submitting."}

        submitted = set(genes)
        expected = set(self.results.loc[self.results.q_value <= 0.05, "gene"])
        tp = len(submitted & expected)
        fp = len(submitted - expected)
        fn = len(expected - submitted)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2*precision*recall/(precision+recall) if precision+recall else 0.0

        process = {
            "inspected_metadata": any(e[0] == "inspect_experiment" for e in self.events),
            "used_valid_normalization": any(e[0] == "normalize_counts" and e[1].get("method") == "log1p_cpm" for e in self.events),
            "used_bh_correction": any(e[0] == "differential_expression" and e[1].get("correction") == "BH" for e in self.events),
        }
        process_score = sum(process.values()) / len(process)
        final_score = 0.75*f1 + 0.25*process_score

        return {
            "score": round(final_score, 3),
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "expected_calls_based_on_analysis": sorted(expected),
            "submitted": sorted(submitted),
            "process_checks": process,
        }

    def call(self, name: str, args: dict):
        if name == "inspect_experiment":
            return self.inspect_experiment()
        if name == "normalize_counts":
            return self.normalize_counts(**args)
        if name == "differential_expression":
            return self.differential_expression(**args)
        if name == "submit":
            return self.submit(**args)
        return {"error": f"Unknown tool: {name}"}
