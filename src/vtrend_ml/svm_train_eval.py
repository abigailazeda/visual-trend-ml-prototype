import argparse, json
from pathlib import Path
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score
from joblib import dump, load

def main(features_npz: str, out_model: str):
    npz = np.load(features_npz)
    X = np.concatenate([npz["hsv"], npz["clip"]], axis=1) if "clip" in npz else npz["hsv"]
    y = npz["y"].astype(int)

    # Simple split (hold-out). You can swap for StratifiedKFold if you want.
    n = len(y)
    idx = np.arange(n)
    np.random.default_rng(42).shuffle(idx)
    split = int(0.8 * n)
    tr, te = idx[:split], idx[split:]
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]

    clf = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),  # sparse-safe if needed
        ("svm", SVC(C=1.0, kernel="rbf", probability=True, class_weight="balanced", random_state=42)),
    ])
    clf.fit(Xtr, ytr)

    yp = clf.predict(Xte)
    yprob = clf.predict_proba(Xte)[:, 1]
    print(classification_report(yte, yp, digits=4))
    try:
        print("ROC-AUC:", roc_auc_score(yte, yprob))
    except ValueError:
        print("ROC-AUC: n/a")

    Path(out_model).parent.mkdir(parents=True, exist_ok=True)
    dump(clf, out_model)
    print("Saved:", out_model)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/processed/features.npz")
    ap.add_argument("--out", default="runs/svm_model.joblib")
    args = ap.parse_args()
    main(args.features, args.out)
