
# Visual Trend ML Prototype

**Predictive modeling of visual trends on social media** using machine learning (CNN, LSTM, SVM) with a reproducible pipeline.

> Author: **Azed Yayah Durrotun Nihayah** · Supervisor: **Agus Priyadi** · License: MIT

## Quickstart
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Download (optional, requires Kaggle auth)
python scripts/download_kaggle.py --dataset "ptx0123/instagram-images-with-captions" --out data/raw

# Organize & build labels
python scripts/organize_data.py --src data/raw --out data/processed --size 224 --meta data/raw/meta.json

# Validate, dedupe, normalize, split
python scripts/validate.py --data data/processed --schema data/schema_labels.json
python scripts/dedupe.py --data data/processed --method phash --threshold 5 --action mark
python scripts/normalize_metadata.py --data data/processed --max_likes 500000 --max_comments 100000
python scripts/split_balance.py --data data/processed --train 0.7 --val 0.15 --seed 42

# Features (HSV + CLIP) → MLP
python scripts/process_features.py --data data/processed --out data/processed/features.npz --device cpu
python -m vtrend_ml.train --config configs/config.yaml
python -m vtrend_ml.evaluate --ckpt runs/best.ckpt --config configs/config.yaml

# Or switch to image mode in configs/config.yaml (model.input_type: image)
```

## Structure
See comments inside `scripts/` and `src/vtrend_ml/` for details.

## Citation
```
@software{visual-trend-ml-prototype,
  author = {Azed Yayah Durrotun Nihayah},
  title  = {Visual Trend ML Prototype},
  year   = {2025}
}
```
