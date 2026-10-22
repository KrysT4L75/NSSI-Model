"""Generic local data interface, validation, and split seeds."""
import argparse
import csv
import json
import numpy as np
import pandas as pd
from .model import CLINICAL_CAT, CLINICAL_NUM, GENETIC
from pathlib import Path


def read_feature_csv(path, id_column="sample_id"):
    """Shared training/inference identity checks before pandas renames headers."""
    with open(path, encoding="utf-8-sig", newline="") as handle:
        headers = next(csv.reader(handle), [])
    if len(headers) != len(set(headers)):
        raise ValueError("Duplicate CSV column names")
    # Preserve literal codes (NA, NULL, 01, etc.) until the schema interprets them.
    X = pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)
    if id_column not in X:
        raise ValueError("CSV requires the configured identifier column")
    X = X.set_index(id_column)
    if not X.index.is_unique or X.index.isna().any() or any(not str(i).strip() for i in X.index):
        raise ValueError("Identifiers must be unique and nonmissing")
    return X


def read_inputs(path, *, clinical=True, expected_n=107):
    X = path.copy() if isinstance(path, pd.DataFrame) else read_feature_csv(path).replace("", np.nan)
    if "label" not in X:
        raise ValueError("CSV requires label")
    X = X.sort_index()  # np.intersect1d in the historical loader sorted IDs.
    y = pd.to_numeric(X.pop("label"), errors="raise")
    if y.isna().any() or not y.isin([0, 1]).all():
        raise ValueError("label must be 0 or 1")
    y = y.astype(int)
    if expected_n is not None and (len(X) != expected_n or (expected_n == 107 and y.value_counts().to_dict() != {0: 53, 1: 54})):
        raise ValueError("The article cohort requires 107 participants: 53 controls, 54 recent-NSSI")
    for prefix in ["met__", "inf__", "blood__"]:
        if not any(c.startswith(prefix) for c in X):
            raise ValueError(f"Missing modality {prefix}")
    required = set(GENETIC) | (set(CLINICAL_NUM + CLINICAL_CAT) if clinical else set())
    if not required.issubset(X.columns):
        raise ValueError(f"Missing required columns: {sorted(required - set(X.columns))}")
    for col in X:
        if col not in CLINICAL_CAT and not col.startswith("blood__") and X[col].isin(["max", "min"]).any():
            raise ValueError(f"Interval codes are allowed only for blood measurements: {col}")
        if col.startswith(("met__", "inf__")) and X[col].isna().any():
            raise ValueError(f"S3 describes complete molecular assays; missing values in {col} need review")
        if col not in CLINICAL_CAT:
            converted = pd.to_numeric(X[col].replace({"max": np.nan, "min": np.nan}), errors="raise")
            if np.isinf(converted).any():
                raise ValueError(f"Infinite value in {col}")
    if clinical:
        for col, categories in zip(CLINICAL_CAT, [{"Female", "Male", "Unknown"}, {"No", "Yes", "Unknown"}, {"No", "Yes", "Unknown"}]):
            if not set(X[col].dropna()).issubset(categories):
                raise ValueError(f"Invalid category in {col}")
    return X, y


def read_seeds(path):
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    seeds = record["seeds"]
    if len(seeds) != 20 or len(set(seeds)) != 20 or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds):
        raise ValueError("Specify exactly 20 unique nonnegative integer outer seeds")
    return seeds, record


def remove_overlapping_blood(X, feature_map):
    """V7 P26/S3: remove blood indicators overlapping molecular modalities.

    feature_map maps every molecular input column to a canonical analyte ID.
    IDs are supplied by the assay dictionary, never inferred from similar names.
    """
    molecular = [c for c in X if c.startswith(("met__", "inf__", "blood__"))]
    if not set(molecular).issubset(feature_map):
        raise ValueError("Feature map must identify every molecular input column")
    if any(not isinstance(feature_map[c], str) or not feature_map[c].strip() for c in molecular):
        raise ValueError("Canonical analyte IDs must be nonempty strings")
    others = {feature_map[c] for c in molecular if c.startswith(("met__", "inf__"))}
    dropped = [c for c in molecular if c.startswith("blood__") and feature_map[c] in others]
    result = X.drop(columns=dropped)
    if not any(c.startswith("blood__") for c in result):
        raise ValueError("No nonoverlapping blood inputs remain")
    return result, dropped


def align_modalities(modalities, labels):
    """Join already QC-processed, unscaled tables; report losses explicitly.

    No duplicate averaging, column deletion or genotype transformation is implicit.
    Inputs must have participant IDs as their index and disjoint feature names.
    """
    for name, frame in {**modalities, "labels": labels}.items():
        if not frame.index.is_unique or frame.index.isna().any():
            raise ValueError(f"Invalid participant IDs in {name}")
    common = labels.index
    for frame in modalities.values():
        common = common.intersection(frame.index)
    common = common.sort_values()
    X = pd.concat([frame.loc[common] for frame in modalities.values()], axis=1)
    if not X.columns.is_unique:
        raise ValueError("Duplicate feature names; resolve overlap explicitly before modeling")
    report = {name: {"available": len(frame), "not_in_intersection": len(frame.index.difference(common))}
              for name, frame in modalities.items()}
    return X, labels.loc[common], report


GENETIC_ROLES = {"polygenic_score": "PRS_EUR", "rare_variant_burden": "Rare_Burden"}


CLINICAL_ROLES = {"age": "clinical__age", "depression_score": "clinical__bdi_ii",
                  "anxiety_score": "clinical__scared", "sex": "clinical__sex",
                  "depression_diagnosis": "clinical__depression",
                  "anxiety_diagnosis": "clinical__anxiety"}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate schema key: {key}")
        result[key] = value
    return result


def load_schema(path):
    schema = json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=_unique_object)
    if schema.get("version") != 1:
        raise ValueError("Unsupported interface version")
    return schema


def adapt_inputs(path, schema, *, outcome=True, required=None):
    """Map external names to model roles without fitting any transformation.

    Molecular feature keys are caller-chosen stable aliases, not source headers.
    The same aliases must be used in training and inference. Extra source columns
    are ignored. Actual schemas, data and generated manifests stay local.
    """
    raw = read_feature_csv(path, id_column=schema["id_column"])
    result = pd.DataFrame(index=raw.index)
    result.index.name = "sample_id"
    analytes, panel = {}, {}
    seen_sources = set()
    molecular_sources, panel_sources = set(), set()

    def assign(target, specification, *, reuse=False):
        if required is not None and target not in required and target != "label":
            return
        if not isinstance(specification, dict) or not isinstance(specification.get("column"), str):
            raise ValueError("Each role needs a column specification")
        source = specification["column"]
        if target != "label" and source == schema.get("outcome", {}).get("column"):
            raise ValueError("Outcome column cannot be used as a predictor")
        if source == schema["id_column"] or (source in seen_sources and not (reuse and source in molecular_sources)):
            raise ValueError("Identifier or repeated source assigned to a model feature")
        if source not in raw:
            raise ValueError(f"Missing configured column for {target}")
        missing = specification.get("missing_values", schema.get("missing_values", [""]))
        if not isinstance(missing, list) or any(not isinstance(v, str) for v in missing):
            raise ValueError("missing_values must be a list of literal strings")
        values = raw[source].mask(raw[source].isin(missing), np.nan)
        if "values" in specification:
            mapping = specification["values"]
            if set(missing) & set(mapping):
                raise ValueError(f"Category and missing-value definitions overlap for {target}")
            keys = values.astype("string")
            unknown = values.notna() & ~keys.isin(mapping)
            if unknown.any():
                raise ValueError(f"Unmapped category for {target}")
            values = keys.map(mapping)
        if target not in CLINICAL_CAT:
            intervals = values.isin(["min", "max"])
            numeric = pd.to_numeric(values.mask(intervals), errors="raise").astype(float)
            values = numeric.astype(object).where(~intervals, values) if intervals.any() else numeric
        else:
            values = values.astype(object).where(values.notna(), np.nan)
        result[target] = values
        seen_sources.add(source)

    if outcome:
        assign("label", schema["outcome"])
    for block, prefix in [("metabolomics", "met__"), ("inflammation", "inf__"), ("blood", "blood__")]:
        entries = schema.get("modalities", {}).get(block, {})
        if not entries:
            raise ValueError(f"Empty modality specification: {block}")
        for alias, specification in sorted(entries.items()):
            if not alias or not isinstance(specification.get("analyte_id"), str) or not specification["analyte_id"].strip():
                raise ValueError("Every molecular alias needs a nonempty analyte_id")
            target = prefix + alias
            assign(target, specification)
            molecular_sources.add(specification["column"])
            analytes[target] = specification["analyte_id"]
    for role, target in GENETIC_ROLES.items():
        assign(target, schema["genetics"][role])
    for role, specification in schema.get("clinical", {}).items():
        if role not in CLINICAL_ROLES:
            raise ValueError(f"Unknown clinical role: {role}")
        assign(CLINICAL_ROLES[role], specification)
    for marker, specification in sorted(schema.get("panel", {}).items()):
        source = specification.get("column")
        if source in panel_sources:
            raise ValueError("Panel markers require distinct source columns")
        nonmolecular = {s.get("column") for s in schema.get("genetics", {}).values()} | {s.get("column") for s in schema.get("clinical", {}).values()}
        if source in nonmolecular:
            raise ValueError("Panel cannot reuse genetic or clinical predictor columns")
        target = "panel__" + marker
        assign(target, specification, reuse=True)
        panel_sources.add(source)
        panel[marker] = target
    if required is not None and not set(required).issubset(result.columns):
        raise ValueError("Schema does not define all inputs consumed by the fitted model")
    return result, analytes, panel or None


def generate_seeds(master_seed=20261021):
    rng = np.random.default_rng(master_seed)
    return [int(s) for s in rng.choice(2**31 - 1, size=20, replace=False)]


def save_seeds(path, master_seed=20261021):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Never silently replace the split contract of an existing experiment.
    with path.open("x", encoding="utf-8") as handle:
        json.dump({"master_seed": master_seed, "seeds": generate_seeds(master_seed),
                   "generator": "numpy.default_rng PCG64; choice without replacement from [0, 2147483647)",
                   "numpy_version": np.__version__, "selection": "generated before fitting; no performance filtering"}, handle, indent=2)



INTERFACE_TEMPLATE = {'version': 1,
 'id_column': 'record_key',
 'outcome': {'column': 'target', 'values': {'negative': 0, 'positive': 1}},
 'modalities': {'metabolomics': {'feature_001': {'column': 'column_a', 'analyte_id': 'analyte_a'}},
                'inflammation': {'feature_001': {'column': 'column_b', 'analyte_id': 'analyte_b'}},
                'blood': {'feature_001': {'column': 'column_c', 'analyte_id': 'analyte_c'}}},
 'genetics': {'polygenic_score': {'column': 'column_d'},
              'rare_variant_burden': {'column': 'column_e'}},
 'clinical': {'age': {'column': 'column_f'},
              'depression_score': {'column': 'column_g'},
              'anxiety_score': {'column': 'column_h'},
              'sex': {'column': 'column_i',
                      'values': {'category_a': 'Female',
                                 'category_b': 'Male',
                                 'unspecified': 'Unknown'}},
              'depression_diagnosis': {'column': 'column_j',
                                       'values': {'absent': 'No',
                                                  'present': 'Yes',
                                                  'unspecified': 'Unknown'}},
              'anxiety_diagnosis': {'column': 'column_k',
                                    'values': {'absent': 'No',
                                               'present': 'Yes',
                                               'unspecified': 'Unknown'}}}}


def main():
    parser = argparse.ArgumentParser(description="Create local interface or random-seed files")
    commands = parser.add_subparsers(dest="command", required=True)
    schema = commands.add_parser("template", help="Write a generic interface template without data")
    schema.add_argument("--output", required=True)
    seeds = commands.add_parser("seeds", help="Generate and save 20 unique random seeds")
    seeds.add_argument("--output", required=True)
    seeds.add_argument("--master-seed", type=int, default=20261021)
    args = parser.parse_args()
    if args.command == "seeds":
        save_seeds(args.output, args.master_seed)
    else:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            json.dump(INTERFACE_TEMPLATE, handle, indent=2)


if __name__ == "__main__":
    main()
