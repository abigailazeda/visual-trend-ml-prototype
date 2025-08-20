import argparse, json, re
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, roc_auc_score

def tokenize(s):
    return re.findall(r"[A-Za-z0-9_#@]+", (s or "").lower())

def build_vocab(captions, min_freq=2, max_size=20000):
    from collections import Counter
    cnt = Counter()
    for c in captions: cnt.update(tokenize(c))
    vocab = {"<pad>":0, "<unk>":1}
    for tok, f in cnt.most_common():
        if f < min_freq or len(vocab) >= max_size: break
        vocab[tok] = len(vocab)
    return vocab

def numericalize(text, vocab, max_len=40):
    ids = [vocab.get(tok, 1) for tok in tokenize(text)]
    ids = ids[:max_len]
    return ids + [0]*(max_len - len(ids))

class CaptionDS(Dataset):
    def __init__(self, labels_path, vocab=None, max_len=40, split=0.8, train=True):
        items = json.loads(Path(labels_path).read_text())
        # keep samples with captions
        items = [it for it in items if (it.get("caption","") is not None)]
        n = len(items)
        idx = np.arange(n); rng = np.random.default_rng(42); rng.shuffle(idx)
        cut = int(split*n)
        self.items = [items[i] for i in (idx[:cut] if train else idx[cut:])]
        self.max_len = max_len
        self.owns_vocab = False
        if vocab is None:
            self.owns_vocab = True
            vocab = build_vocab([it.get("caption","") for it in self.items])
        self.vocab = vocab
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        it = self.items[i]
        x = torch.tensor(numericalize(it.get("caption",""), self.vocab, self.max_len), dtype=torch.long)
        y = torch.tensor(float(it.get("label",0)), dtype=torch.float32)
        return x, y

class LSTMText(nn.Module):
    def __init__(self, vocab_size, emb=128, hidden=128, layers=1, dropout=0.2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))
    def forward(self, x):
        e = self.emb(x)
        o, (h, c) = self.lstm(e)
        return self.fc(h[-1])

def main(labels_json, epochs, bs, lr, ckpt):
    # Build train set (also builds vocab), then test set uses same vocab
    ds_tr = CaptionDS(labels_json, vocab=None, train=True)
    ds_te = CaptionDS(labels_json, vocab=ds_tr.vocab, train=False)
    vocab_size = len(ds_tr.vocab)
    print("Vocab size:", vocab_size, "- train:", len(ds_tr), "test:", len(ds_te))

    dl_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True)
    dl_te = DataLoader(ds_te, batch_size=bs, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LSTMText(vocab_size=vocab_size).to(device)
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

    torch.save({"state_dict": model.state_dict(), "vocab": ds_tr.vocab}, ckpt)
    print("Saved:", ckpt)

    # Eval
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for xb, yb in dl_te:
            prob = torch.sigmoid(model(xb.to(device))).cpu().numpy().ravel()
            ys += yb.numpy().tolist()
            ps += prob.tolist()
    import numpy as np
    preds = (np.array(ps) >= 0.5).astype(int)
    print(classification_report(ys, preds, digits=4))
    try:
        print("ROC-AUC:", roc_auc_score(ys, ps))
    except ValueError:
        print("ROC-AUC: n/a")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/processed/labels.json")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ckpt", default="runs/lstm_text.ckpt")
    main(**vars(ap.parse_args()))


run
python -m vtrend_ml.lstm_text_train_eval --labels data/processed/labels.json --epochs 5 --ckpt runs/lstm_text.ckpt
