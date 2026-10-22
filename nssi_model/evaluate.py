"""Repeated holdouts, benchmarks, ablations, and statistical summaries."""
import argparse
import hashlib
import joblib
import json
import numpy as np
import pandas as pd
import platform
import scipy
import sklearn
import warnings
from .data import adapt_inputs, load_schema, read_inputs, read_seeds, remove_overlapping_blood
from .model import IntervalCodes, UniversalStacking, clinical_model, experts, genomic_model, logistic, panel_model
from collections import Counter
from pathlib import Path
from scipy.stats import false_discovery_control, mannwhitneyu, t, wilcoxon
from sklearn.base import clone
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split


def selected_features(model):
    selected = {}
    for name, pipe in zip(model.base_names_, model.fitted_base_models_):
        if name == "Clinical":
            continue
        names = np.asarray(pipe.named_steps["columns"].columns_)
        if "screen" in pipe.named_steps:
            mask = pipe.named_steps["screen"].get_support()
        elif name.endswith("_EN"):
            mask = pipe.named_steps["search"].best_estimator_.named_steps["model"].coef_[0] != 0
        else:
            # Fig. 2a passes the unscreened blood branch directly to its RF.
            mask = np.ones(len(names), dtype=bool)
        selected[name] = names[mask].tolist()
    return selected


def selection_table(feature_names, selected_per_run):
    """S_j = sum_r I(j in S_r) / R; count once per feature per run."""
    rows = []
    sets = [set(run) for run in selected_per_run]
    if not sets:
        raise ValueError("No completed feature-selection runs")
    for feature in sorted(feature_names):
        count = sum(feature in run for run in sets)
        rows.append({"feature": feature, "count": count,
                     "frequency": count / len(sets), "stable": count >= np.ceil(.8 * len(sets))})
    return pd.DataFrame(rows)


def coefficient_group(term):
    if term.endswith("_x_PRS_EUR"):
        return "PRS × Omics interaction"
    if term.endswith("_x_Rare_Burden"):
        return "Rare burden × Omics interaction"
    if term == "PRS_EUR":
        return "PRS"
    if term == "Rare_Burden":
        return "Rare burden"
    if term.startswith("Met_"):
        return "Metabolism"
    if term.startswith("Inf_"):
        return "Inflammation"
    if term == "Blood_RF":
        return "Blood baseline"
    raise ValueError(f"Unknown Omics coefficient term: {term}")


def coefficient_shares(coefficients):
    means = coefficients.assign(abs_coef=coefficients.coefficient.abs()).groupby("term").abs_coef.mean()
    groups = means.groupby([coefficient_group(term) for term in means.index]).sum()
    if groups.sum() == 0:
        raise ValueError("Coefficient shares undefined when all coefficients are zero")
    return (100 * groups / groups.sum()).rename("relative_absolute_coefficient_percent")


def stable_feature_tests(X, y, features):
    """S5 descriptive, cohort-derived MWU/BH family; no effect on fitting."""
    rows = []
    for feature in features:
        values = pd.to_numeric(X[feature], errors="raise")
        a, b = values[y == 0].dropna(), values[y == 1].dropna()
        if not len(a) or not len(b):
            raise ValueError(f"No observations for a stable-feature comparison: {feature}")
        rows.append({"feature": feature, "n_non_nssi": len(a), "n_nssi": len(b),
                     "p_value": mannwhitneyu(a, b, alternative="two-sided").pvalue})
    result = pd.DataFrame(rows, columns=["feature", "n_non_nssi", "n_nssi", "p_value"])
    result["q_value"] = false_discovery_control(result.p_value.to_numpy(), method="bh") if len(result) else []
    return result


METRIC_NAMES = ["AUC", "AUPRC", "Brier", "Accuracy", "Balanced_accuracy",
                "Sensitivity", "Specificity", "PPV", "NPV", "F1"]


def metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    if not len(y):
        return {"n": 0, "n_positive": 0, **dict.fromkeys(METRIC_NAMES, np.nan)}
    tn, fp, fn, tp = confusion_matrix(y, p >= 0.5, labels=[0, 1]).ravel()
    both_classes = len(np.unique(y)) == 2
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    return {"n": len(y), "n_positive": int(y.sum()),
            "AUC": roc_auc_score(y, p) if both_classes else np.nan,
            # Non-interpolated precision-recall area (average precision).
            "AUPRC": average_precision_score(y, p) if both_classes else np.nan,
            "Brier": brier_score_loss(y, p) if len(y) else np.nan,
            "Accuracy": (tn + tp) / len(y) if len(y) else np.nan,
            "Balanced_accuracy": (sensitivity + specificity) / 2 if both_classes else np.nan,
            "Sensitivity": sensitivity, "Specificity": specificity,
            "PPV": tp / (tp + fp) if tp + fp else np.nan,
            "NPV": tn / (tn + fn) if tn + fn else np.nan,
            "F1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else np.nan}


def summarize(scores):
    rows = []
    for (model, subset), group in scores.groupby(["model", "subset"]):
        for metric in METRIC_NAMES:
            v = group[metric].dropna()
            mean, sd = v.mean(), v.std(ddof=1)
            half = t.ppf(0.975, len(v) - 1) * sd / np.sqrt(len(v)) if len(v) > 1 else np.nan
            rows.append({"model": model, "subset": subset, "metric": metric, "n_splits": len(v),
                         "mean": mean, "sd": sd, "variance": v.var(ddof=1),
                         "ci_low": mean-half, "ci_high": mean+half})
    return pd.DataFrame(rows)


def paired_comparisons(scores):
    full = scores.query("model == 'Omics' and subset == 'complete'").set_index("seed")
    rows = []
    for name, group in scores.query("subset == 'complete' and model != 'Omics'").groupby("model"):
        other = group.set_index("seed")
        if set(other.index) != set(full.index):
            raise ValueError("Paired model splits do not match")
        other = other.loc[full.index]
        for metric in ["AUC", "Brier"]:
            diff = full[metric] - other[metric]
            if not np.isfinite(diff).all():
                raise ValueError("Undefined score in complete-test paired comparison")
            p = 1.0 if np.all(diff == 0) else wilcoxon(diff, alternative="greater" if metric == "AUC" else "less").pvalue
            rows.append({"reference": "Omics", "comparison": name, "metric": metric,
                         "mean_reference_minus_comparison": diff.mean(), "p_one_sided": p, "n_pairs": len(diff)})
    return pd.DataFrame(rows)


def evaluate(X, y, seeds, output, *, clinical=True, ablations=True, save_models=False, panel_columns=None, software_test=False):
    if not X.index.equals(y.index):
        raise ValueError("X/y participant order differs")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be nonempty and unique")
    if not software_test and len(seeds) != 20:
        raise ValueError("Paper evaluation requires all 20 splits; use software_test=True only for artificial checks")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    scores, predictions, splits, folds, coefficients = [], [], [], [], []
    feature_counts, branch_counts = Counter(), Counter()
    selection_runs, selection_rows, oof_rows, tuned_parameters = [], [], [], []
    base = experts()
    for seed in seeds:
        (output / "progress.json").write_text(json.dumps({"seed": seed, "status": "running"}), encoding="utf-8")
        tr, te = train_test_split(np.arange(len(X)), test_size=0.2, stratify=y, random_state=seed)
        train, test = X.iloc[tr], X.iloc[te]
        ytrain, ytest = y.iloc[tr], y.iloc[te]
        if len(X) == 107 and ytest.value_counts().to_dict() != {0: 11, 1: 11}:
            raise AssertionError("Expected 11 participants per group in test set")
        splits.extend({"seed": seed, "sample_id": str(i), "partition": part}
                      for part, ids in [("train", train.index), ("test", test.index)] for i in ids)
        pd.DataFrame(splits).to_csv(output / "outer_splits_partial.csv", index=False)
        omics = UniversalStacking(base, logistic()).fit(train, ytrain)
        fitted = {"Omics": omics, "Geno_LR": genomic_model(random_state=seed).fit(train, ytrain)}
        fitted.update(zip(omics.base_names_, omics.fitted_base_models_))
        if panel_columns is not None:
            fitted["Seven-marker panel"] = panel_model(panel_columns).fit(train, ytrain)
        if clinical:
            fitted["Clinical"] = clinical_model().fit(train, ytrain)
            fitted["Clinical + Omics"] = UniversalStacking(base + [("Clinical", clinical_model())], logistic()).fit(train, ytrain)
        if ablations:
            for label, prefix in [("Metabolomics", "Met_"), ("Inflammation", "Inf_"), ("Blood", "Blood_")]:
                fitted["w/o " + label] = UniversalStacking([(n, e) for n, e in base if not n.startswith(prefix)], logistic()).fit(train, ytrain)
            fitted["w/o Genomics"] = UniversalStacking(base, logistic(), use_genomics=False, use_interaction=False).fit(train, ytrain)
        for name, model in fitted.items():
            p = model.predict_proba(test)[:, 1]
            scores.append({"seed": seed, "model": name, "subset": "complete", **metrics(ytest, p)})
            predictions.extend({"seed": seed, "model": name, "sample_id": str(i), "label": int(a), "probability": float(b)}
                               for i, a, b in zip(test.index, ytest, p))
            if name == "Omics" and "clinical__depression" in test:
                mask = (test["clinical__depression"] == "Yes").to_numpy()
                scores.append({"seed": seed, "model": name, "subset": "depression", **metrics(ytest.to_numpy()[mask], p[mask])})
            if isinstance(model, UniversalStacking):
                folds.extend({"seed": seed, "model": name, **f} for f in model.fold_audit_)
                for row_index, participant in enumerate(train.index):
                    oof_rows.append({"seed": seed, "model": name, "sample_id": str(participant),
                                     "inner_fold": int(model.oof_fold_[row_index]),
                                     **dict(zip(model.meta_feature_names_, model.oof_meta_features_[row_index]))})
        selected = selected_features(omics)
        selection_runs.append(set(f for fs in selected.values() for f in fs))
        feature_counts.update(set(f for fs in selected.values() for f in fs))
        for branch, fs in selected.items():
            branch_counts.update((branch, f) for f in fs)
            selection_rows.extend({"seed": seed, "branch": branch, "feature": f} for f in fs)
        for name, pipe in zip(omics.base_names_, omics.fitted_base_models_):
            if "search" in pipe.named_steps:
                tuned_parameters.append({"seed": seed, "expert": name, **pipe.named_steps["search"].best_params_})
        coefficients.extend({"seed": seed, "term": term, "coefficient": float(coef)}
                            for term, coef in zip(omics.meta_feature_names_, omics.fitted_meta_model_.coef_[0]))
        if save_models:
            # Fitted objects include training-derived state and IDs; keep them private.
            joblib.dump(omics, output / f"omics_seed_{seed}.joblib")
        print(f"Finished seed {seed}", flush=True)
        pd.DataFrame(scores).to_csv(output / "metrics_partial.csv", index=False)
        # Checkpoint every completed seed before attempting the next one.
        for filename, rows in [("test_predictions", predictions), ("oof_meta_inputs", oof_rows),
                               ("selected_features_per_run", selection_rows),
                               ("selected_hyperparameters", tuned_parameters),
                               ("meta_coefficients", coefficients)]:
            pd.DataFrame(rows).to_csv(output / f"{filename}_partial.csv", index=False)
        (output / "inner_folds_partial.json").write_text(json.dumps(folds, indent=2), encoding="utf-8")
        (output / "completed_seeds.json").write_text(json.dumps(seeds[:len(selection_runs)], indent=2), encoding="utf-8")
        (output / "progress.json").write_text(json.dumps({"seed": seed, "status": "seed_completed"}), encoding="utf-8")
    score_df = pd.DataFrame(scores)
    score_df.to_csv(output / "metrics.csv", index=False)
    subgroup = score_df.query("model == 'Omics' and subset == 'depression'").set_index("seed")
    if len(subgroup):
        complete = score_df.query("model == 'Omics' and subset == 'complete'").set_index("seed")
        deltas = subgroup[METRIC_NAMES] - complete.loc[subgroup.index, METRIC_NAMES]
        deltas.add_prefix("depression_minus_complete_").to_csv(output / "depression_paired_deltas.csv")
        summarize(deltas.assign(model="Omics", subset="depression_minus_complete")).to_csv(
            output / "depression_paired_delta_summary.csv", index=False)
    summarize(score_df).to_csv(output / "summary.csv", index=False)
    paired_comparisons(score_df).to_csv(output / "paired_comparisons.csv", index=False)
    pd.DataFrame(predictions).to_csv(output / "test_predictions.csv", index=False)
    pd.DataFrame(splits).to_csv(output / "outer_splits.csv", index=False)
    (output / "inner_folds.json").write_text(json.dumps(folds, indent=2), encoding="utf-8")
    molecular_columns = [c for c in X if c.startswith(("met__", "inf__", "blood__"))]
    stable = selection_table(molecular_columns, selection_runs)
    stable.to_csv(output / "selection_frequencies.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(output / "selected_features_per_run.csv", index=False)
    pd.DataFrame(oof_rows).to_csv(output / "oof_meta_inputs.csv", index=False)
    pd.DataFrame(tuned_parameters).to_csv(output / "selected_hyperparameters.csv", index=False)
    pd.DataFrame([{"branch": b, "feature": f, "count": n, "frequency": n / len(seeds)} for (b, f), n in sorted(branch_counts.items())]).to_csv(output / "branch_selection_frequencies.csv", index=False)
    coef = pd.DataFrame(coefficients)
    coef.to_csv(output / "meta_coefficients.csv", index=False)
    coefficient_shares(coef).to_csv(output / "coefficient_shares.csv")
    # Univariate inference uses observed values; coded blood extrema are resolved
    # descriptively only after all predictive fits, never fed back into a model.
    descriptive = pd.DataFrame(IntervalCodes().fit_transform(X[molecular_columns]), columns=molecular_columns, index=X.index)
    stable_feature_tests(descriptive, y, stable.loc[stable.stable, "feature"].tolist()).to_csv(output / "stable_feature_mwu_bh.csv", index=False)
    (output / "completed.json").write_text(json.dumps({"seeds": seeds, "n_participants": len(X), "stable_feature_count": int(stable.stable.sum()), "software_test": software_test, "manuscript_numerical_agreement": "not assessed"}, indent=2), encoding="utf-8")
    return score_df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--schema", required=True, help="Local JSON mapping caller-defined columns to model roles")
    parser.add_argument("--output", required=True)
    parser.add_argument("--omics-only", action="store_true")
    parser.add_argument("--no-ablations", action="store_true")
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--panel", action="store_true", help="Enable the seven-marker mapping in the local schema")
    args = parser.parse_args()
    schema = load_schema(args.schema)
    adapted, feature_mapping, configured_panel = adapt_inputs(args.data, schema)
    X, y = read_inputs(adapted, clinical=not args.omics_only)
    X, removed = remove_overlapping_blood(X, feature_mapping)
    seeds, seed_record = read_seeds(args.seeds)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    panel_mapping = configured_panel if args.panel else None
    if args.panel and panel_mapping is None:
        raise ValueError("--panel requires a seven-marker schema mapping")
    manifest = {"python": platform.python_version(), "sklearn": sklearn.__version__,
                "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
                "input_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
                "seed_contract": seed_record, "threshold": 0.5, "removed_overlapping_blood": removed,
                "schema_sha256": hashlib.sha256(Path(args.schema).read_bytes()).hexdigest(), "schema": schema,
                "feature_mapping": feature_mapping,
                "options": vars(args), "panel_mapping": panel_mapping,
                "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob("*.py")}}
    status = {"status": "failed", "error": None}
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            evaluate(X, y, seeds, args.output, clinical=not args.omics_only,
                     ablations=not args.no_ablations, save_models=args.save_models,
                     panel_columns=panel_mapping)
            status["status"] = "completed"
        except Exception as exc:
            status["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            output.mkdir(parents=True, exist_ok=True)
            convergence = sum(issubclass(w.category, sklearn.exceptions.ConvergenceWarning) for w in caught)
            status["convergence_warnings"] = convergence
            if status["status"] == "completed" and convergence:
                status["status"] = "completed_with_convergence_warnings"
            (output / "run_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
            (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            (output / "warnings.txt").write_text("\n".join(f"{w.category.__name__}: {w.message}" for w in caught), encoding="utf-8")


if __name__ == "__main__":
    main()
