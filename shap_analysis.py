"""
SHAP-based explainability on the Branch-A (handcrafted) + Branch-B (deep)
hybrid feature space, using the project's own data-loading and feature
functions (experiments/common.py) so results are consistent with the real
pipeline rather than re-implemented assumptions.

NOTE: this deliberately runs only the ML base classifiers (RF, XGBoost) on
the FULL hybrid feature set (no Stage-I feature selection, no DL branch,
no hierarchical TLBO) to keep this fast and self-contained. State this
scope explicitly in the paper.

Usage:
    python scripts/shap_analysis.py --dataset uc_merced --run 0 --config configs/<your_config>.yaml
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.common import load_and_split, build_feature_matrices
from src.models.base_classifiers import fit_predict_proba
from src.utils.io import load_config
from src.utils.seed import set_seed

import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="uc_merced")
    parser.add_argument("--run", type=int, default=0)
    parser.add_argument("--config", required=True)
    parser.add_argument("--outdir", default="results/shap")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    cfg = load_config(args.config)
    seed = cfg["project"]["seed"] + args.run
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}, seed={seed}")

    print("Loading & splitting dataset...")
    ds, idx_train, idx_val, idx_test = load_and_split(args.dataset, cfg, seed)
    num_classes = ds.num_classes
    class_names = getattr(ds, "classes", None) or getattr(ds, "class_names", None)
    if class_names is None:
        raise AttributeError(
            "Could not find class name list on LULCDataset (tried .classes, .class_names). "
            "Check src/data/dataset.py for the correct attribute and edit this line."
        )
    class_names = list(class_names)

    print("Extracting Branch A (handcrafted) + Branch B (deep) hybrid features...")
    feats = build_feature_matrices(ds, idx_train, idx_val, idx_test, cfg, seed, device=device)

    X_train, y_train = feats["X_train"], feats["y_train"]
    X_test, y_test = feats["X_test"], feats["y_test"]
    hc_dim = feats["hc_dim"]
    deep_dim = feats["deep_dim"]
    print(f"Feature dims -> handcrafted: {hc_dim}, deep: {deep_dim}, total: {hc_dim + deep_dim}")

    feature_names = [f"handcrafted_{i}" for i in range(hc_dim)] + [f"deep_{i}" for i in range(deep_dim)]

    print("Training RF and XGBoost on the full hybrid feature set...")
    ml_names = [n for n in cfg["base_classifiers"]["ml"] if n.lower() in ("rf", "random_forest", "xgboost", "xgb")]
    if not ml_names:
        ml_names = cfg["base_classifiers"]["ml"][:2]  # fallback: first two configured ML classifiers
    print(f"Using classifiers: {ml_names}")

    trained = {}
    for name in ml_names:
        clf, val_proba = fit_predict_proba(name, X_train, y_train, feats["X_val"], seed, num_classes)
        trained[name] = clf

    print("Computing SHAP values (TreeExplainer)...")
    shap_results = {}
    for name, clf in trained.items():
        try:
            explainer = shap.TreeExplainer(clf)
            sv = explainer.shap_values(X_test)
            shap_results[name] = sv
            print(f"  done: {name}")
        except Exception as e:
            print(f"  SKIPPED {name}: TreeExplainer not compatible ({e})")

    def mean_abs_shap(sv):
        if isinstance(sv, list):
            return np.mean(np.abs(np.stack(sv, axis=0)), axis=(0, 1))
        return np.mean(np.abs(sv), axis=0)

    df_imp = pd.DataFrame({"feature": feature_names})
    df_imp["group"] = np.where(df_imp["feature"].str.startswith("handcrafted"), "Handcrafted", "Deep")

    for name, sv in shap_results.items():
        df_imp[f"shap_{name}"] = mean_abs_shap(sv)

    df_imp.to_csv(f"{args.outdir}/feature_importance_all.csv", index=False)

    shap_cols = [c for c in df_imp.columns if c.startswith("shap_")]
    group_summary = df_imp.groupby("group")[shap_cols].sum()
    group_pct = 100 * group_summary / group_summary.sum()
    group_pct.to_csv(f"{args.outdir}/handcrafted_vs_deep_contribution.csv")
    print("\n=== Handcrafted vs. Deep feature contribution (% of total |SHAP|) ===")
    print(group_pct)

    first_col = shap_cols[0]
    top20 = df_imp.sort_values(first_col, ascending=False).head(20)
    top20.to_csv(f"{args.outdir}/top20_features.csv", index=False)
    print(f"\n=== Top 20 features (by {first_col}) ===")
    print(top20[["feature", "group", first_col]])

    plt.figure(figsize=(6, 4))
    group_pct[first_col].plot(kind="bar", color=["#4C72B0", "#DD8452"])
    plt.ylabel(f"% of total |SHAP| ({first_col})")
    plt.title("Handcrafted vs. Deep feature contribution")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{args.outdir}/group_contribution_bar.png", dpi=150)
    plt.close()

    confused_classes = [
        "buildings", "denseresidential", "mediumresidential",
        "intersection", "mobilehomepark", "storagetanks",
    ]
    main_name = list(shap_results.keys())[0]
    sv_main = shap_results[main_name]
    if isinstance(sv_main, list):
        for cname in confused_classes:
            if cname not in class_names:
                print(f"WARNING: class '{cname}' not in dataset class list, skipping")
                continue
            cidx = class_names.index(cname)
            sv_c = sv_main[cidx]
            mean_abs = np.mean(np.abs(sv_c), axis=0)
            top10_idx = np.argsort(mean_abs)[::-1][:10]

            plt.figure(figsize=(6, 4))
            plt.barh([feature_names[i] for i in top10_idx][::-1], mean_abs[top10_idx][::-1])
            plt.title(f"Top SHAP features — {cname} ({main_name})")
            plt.xlabel("mean |SHAP value|")
            plt.tight_layout()
            plt.savefig(f"{args.outdir}/shap_{cname}.png", dpi=150)
            plt.close()
            print(f"Saved per-class SHAP plot: {cname}")
    else:
        print("Binary/regression-style SHAP output detected; skipping per-class breakdown.")

    print(f"\nAll outputs saved to {args.outdir}/")


if __name__ == "__main__":
    main()
