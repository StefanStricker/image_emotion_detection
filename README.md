# Image Emotion Detection

Classifies a face into one of 7 basic emotions: `angry`, `disgusted`, `fearful`, `happy`,
`neutral`, `sad`, `surprised`.

## Approach

- **Model**: ConvNeXt-Small (ImageNet-pretrained), fine-tuned in two stages — classifier
  head only, then the full network with a differential learning rate. See
  `notebooks/model_train.ipynb` for the full training setup (focal loss, class-weighted
  sampling, MixUp/CutMix, OneCycle + cosine LR schedules, early stopping)
- **Dataset**: [RAF-DB](http://www.whdeng.cn/raf/model1.html) (aligned, 100×100 RGB,
  real-world photos) — not included in this repo, see [Data setup](#data-setup) below.
- **Serving**: a small FastAPI app (`notebooks/serve.py`) exposing a `/predict` endpoint.

## Results (RAF-DB test set, with test-time augmentation)

| Metric | Value |
|---|---|
| Accuracy | 81.6% |
| Macro F1 | 0.746 |
| Expected Calibration Error (ECE) | 0.168 |

| Class | Precision | Recall | F1 |
|---|---|---|---|
| angry | 0.70 | 0.82 | 0.75 |
| disgusted | 0.50 | 0.66 | 0.57 |
| fearful | 0.51 | 0.62 | 0.56 |
| happy | 0.98 | 0.82 | 0.90 |
| neutral | 0.79 | 0.79 | 0.79 |
| sad | 0.79 | 0.87 | 0.83 |
| surprised | 0.78 | 0.88 | 0.83 |

`fearful` and `disgusted` are the weakest classes — they're also RAF-DB's smallest
classes by far. Full breakdown, confusion matrix, ROC curves, and calibration plots are
in `notebooks/evaluation.ipynb` and `images/*_rafdb.png`.

**Calibration caveat**: confidence scores are not fully trustworthy — the model is
noticeably overconfident in the mid-confidence range (see `images/calibration_rafdb.png`).
Treat the `scores` field from the API as a ranking, not a true probability.

## Project structure

```
notebooks/
  preprocessing.ipynb   # builds data/dataset_split_rafdb/{train,val,test}/<class>/ from the raw Kaggle download
  model_train.ipynb     # trains ConvNeXt-Small, saves convnext_small_rafdb.pth
  evaluation.ipynb      # metrics, confusion matrix, ROC/AUC, calibration, saved to images/
  serve.py              # FastAPI inference server
images/                 # evaluation plots (checked into git)
requirements.txt
```

`data/` and the trained `.pth` weights are gitignored (large/derived — see below for how
to regenerate them).

## Data setup

1. Download RAF-DB (aligned) from Kaggle, e.g.
   [shuvoalok/raf-db-dataset](https://www.kaggle.com/datasets/shuvoalok/raf-db-dataset)
2. Extract data.
3. Run `notebooks/preprocessing.ipynb` — it maps RAF-DB's numeric labels to emotion names,
   carves a validation split out of the official train split, and writes everything to
   `data/dataset_split_rafdb/`.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

1. **Preprocess**: run `notebooks/preprocessing.ipynb`.
2. **Train**: run `notebooks/model_train.ipynb` → produces `convnext_small_rafdb.pth` in
   the project root.
3. **Evaluate**: run `notebooks/evaluation.ipynb` → metrics and plots in `images/`.
4. **Serve**:
   ```bash
   uvicorn notebooks.serve:app --app-dir . --reload
   ```
   ```bash
   curl -X POST "http://127.0.0.1:8000/predict?tta=true" -F "file=@path/to/face.jpg"
   ```
   ```json
   {
     "emotion": "happy",
     "confidence": 0.91,
     "scores": {"angry": 0.01, "disgusted": 0.00, "fearful": 0.01, "happy": 0.91,
                "neutral": 0.04, "sad": 0.01, "surprised": 0.02}
   }
   ```

