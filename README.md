# SHAP-Based Explainability of a Hybrid Handcrafted–Deep Feature Space for LULC Classification

Code and data accompanying the paper:

> Arukonda, S., Chetana, V. L., & Srilakshmi, V. (2026).
> "SHAP-Based Explainability of a Hybrid Handcrafted–Deep Feature Space
> for Land Use and Land Cover Classification."
> *Procedia Computer Science*, 5th ICMLDE 2026.

---

## Files

| File | Description |
|------|-------------|
| `shap_analysis.py` | Main SHAP analysis script — computes TreeExplainer attributions for RF and XGBoost on the 4370-d hybrid feature vector, generates per-class SHAP bar plots and group contribution figures |
| `R22_R24_final.ipynb` | Kaggle notebook — self-contained, runs feature extraction (handcrafted + deep) and the two reviewer-requested experiments: ablation study (R2.2) and multi-seed SHAP consistency (R2.4) |
| `feature_importance_all.csv` | Mean absolute SHAP value per feature (all 4370 features) for RF and XGBoost |
| `handcrafted_vs_deep_contribution.csv` | Group-level SHAP attribution (%) — handcrafted vs. deep, for both classifiers |
| `top20_features.csv` | Top-20 features by mean absolute SHAP value |

---

## Dataset

UC Merced Land Use dataset (Yang & Newsam, 2010) — 21 classes, 2100 images, 0.3 m/pixel.
Available at: http://weegee.vision.ucmerced.edu/datasets/landuse.html

---

## Requirements

```
numpy, pandas, scikit-learn, xgboost, shap, torch, torchvision,
scikit-image, opencv-python, matplotlib, tqdm
```

Install via:
```bash
pip install numpy pandas scikit-learn xgboost shap torch torchvision scikit-image opencv-python matplotlib tqdm
```

---

## Usage

The Kaggle notebook `R22_R24_final.ipynb` is fully self-contained.
Upload it to Kaggle, attach the UC Merced dataset, enable GPU, and run all cells.

For the SHAP analysis script, set paths to your config and dataset as described in the script's argparse help:
```bash
python shap_analysis.py --dataset uc_merced --config configs/your_config.yaml --outdir results/shap
```
