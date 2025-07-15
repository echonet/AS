from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np
import pandas as pd
from pathlib import Path

PRED_DIR = Path("./predictions").resolve()
models = ["PLAX", "PSAX", "Apical", "PLAX_D", "PSAX_D", "Apical_D"]
phenotype = 'AS_severity'
logit_cols = [
    "no_preds", "mild_preds", "mild~moderate_preds",
    "moderate_preds", "moderate~severe_preds", "severe_preds",
]
true_map = {0: 0, 1: 1, 2: 2, 3: 2, 4: 3, 5: 3} # Maps 6-class to 4-class labels

def softmax(x):
    x = np.clip(x, -100, 100)
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

def per_auc(y, p, n):
    return tuple(
        roc_auc_score((y == i).astype(int), p[:, i]) if ((y == i).any() and (y != i).any()) else np.nan
        for i in range(n)
    )

def metrics(y, p, n):
    out = {
        "n": len(y),
        "acc": accuracy_score(y, p.argmax(1)),
        "f1": f1_score(y, p.argmax(1), average="macro"),
    }
    try:
        out["auc_macro"] = roc_auc_score(y, p, multi_class="ovr", average="macro")
    except ValueError:
        out["auc_macro"] = np.nan
    for i, a in enumerate(per_auc(y, p, n)):
        out[f"auc_cls{i}"] = a
    return out

def collapse4(p):
    out = np.empty((p.shape[0], 4), dtype=p.dtype)
    out[:, 0] = p[:, 0]
    out[:, 1] = p[:, 1]
    out[:, 2] = p[:, 2] + p[:, 3]
    out[:, 3] = p[:, 4] + p[:, 5]
    return out

def img_metrics(df, n):
    p6 = softmax(df[logit_cols].to_numpy())
    p = collapse4(p6) if n == 4 else p6
    y = df[phenotype].map(true_map).to_numpy() if n == 4 else df[phenotype].to_numpy()
    return metrics(y, p, n)

def study_metrics(df, n):
    gb = df.groupby("study_uid")
    p6 = softmax(gb[logit_cols].mean().to_numpy())
    p = collapse4(p6) if n == 4 else p6
    y6 = gb[phenotype].max().to_numpy()
    y = np.vectorize(true_map.get)(y6) if n == 4 else y6
    return metrics(y, p, n)

def model_summary(models):
    rows = []
    for model in models:
        f = PRED_DIR / f"{model}.csv"
        if not f.exists():
            print("[WARN]", f, "missing"); continue
        df = pd.read_csv(f, low_memory=False).drop_duplicates()
        for n in (6, 4):
            rows += [
                {"model": model.replace(phenotype+'_', ''), "level": "image", "n_cls": n, **img_metrics(df, n)},
                {"model": model.replace(phenotype+'_', ''), "level": "study", "n_cls": n, **study_metrics(df, n)},
            ]
    return pd.DataFrame(rows)

if __name__ == "__main__":
    # Example usage
    print("Model Summary:")
    summary_df = model_summary(models)
    print(summary_df)
    summary_df.to_csv(PRED_DIR / "model_summary.csv", index=False)