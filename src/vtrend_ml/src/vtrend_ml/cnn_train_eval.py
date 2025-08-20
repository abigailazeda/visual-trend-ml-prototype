import argparse, yaml, torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np
from .data import VisualTrendDataset

def build_model(dropout=0.2):
    m = models.resnet18(weights=None)
    in_f = m.fc.in_features
    m.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_f, 1))
    return m

def main(processed_dir: str, img_size: int, epochs: int, bs: int, lr: float, ckpt: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tfm = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])

    ds = VisualTrendDataset(processed_dir, transform=tfm)
    n = len(ds); idx = np.arange(n); np.random.default_rng(42).shuffle(idx)
    split = int(0.8*n)
    tr_idx, te_idx = idx[:split], idx[split:]
    tr = torch.utils.data.Subset(ds, tr_idx)
    te = torch.utils.data.Subset(ds, te_idx)

    dl_tr = DataLoader(tr, batch_size=bs, shuffle=True)
    dl_te = DataLoader(te, batch_size=bs, shuffle=False)

    model = build_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    model.train()
    for e in range(epochs):
        tot = 0.0
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device).unsqueeze(1)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward(); opt.step()
            tot += loss.item()
        print(f"Epoch {e+1}/{epochs} - loss: {tot/len(dl_tr):.4f}")

    torch.save(model.state_dict(), ckpt)
    print("Saved:", ckpt)

    # Evaluate
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for xb, yb in dl_te:
            prob = torch.sigmoid(model(xb.to(device))).cpu().numpy().ravel()
            ys += yb.numpy().tolist()
            ps += prob.tolist()

    preds = (np.array(ps) >= 0.5).astype(int)
    print(classification_report(ys, preds, digits=4))
    try:
        print("ROC-AUC:", roc_auc_score(ys, ps))
    except ValueError:
        print("ROC-AUC: n/a")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/processed")
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ckpt", default="runs/cnn_best.ckpt")
    main(**vars(ap.parse_args()))

run
python -m vtrend_ml.cnn_train_eval --data data/processed --epochs 5 --ckpt runs/cnn_best.ckpt
