# Image Emotion Detection

Classifies a face into one of 7 basic emotions: `angry`, `disgusted`, `fearful`, `happy`,
`neutral`, `sad`, `surprised`.

## Approach

- **Model**: ConvNeXt-Small (ImageNet-pretrained), fine-tuned in two stages: 1. classifier
  head only, 2. the full network with a differential learning rate. See
  `notebooks/model_train.ipynb` for the full training setup (focal loss, class-weighted
  sampling, CutMix, OneCycle + cosine LR schedules, early stopping)
- **Dataset**: [RAF-DB](http://www.whdeng.cn/raf/model1.html) (aligned, 100×100 RGB,
  real-world photos) not included in this repo, see [Data setup](#data-setup) below.
- **Serving**: a small FastAPI app (`notebooks/serve.py`) exposing a `/predict` endpoint,
  with a Streamlit GUI (`notebooks/streamlit_app.py`) on top for non-technical users.

## Results (RAF-DB test set, with test-time augmentation)

| Metric | Value |
|---|---|
| Accuracy | 86.1% |
| Macro F1 | 0.778 |
| Expected Calibration Error (ECE) | 0.027 |

| Class | Precision | Recall | F1 |
|---|---|---|---|
| angry | 0.83 | 0.75 | 0.79 |
| disgusted | 0.63 | 0.59 | 0.61 |
| fearful | 0.71 | 0.49 | 0.58 |
| happy | 0.96 | 0.93 | 0.95 |
| neutral | 0.82 | 0.86 | 0.84 |
| sad | 0.83 | 0.85 | 0.84 |
| surprised | 0.82 | 0.88 | 0.85 |

`fearful` and `disgusted` are the weakest classes, they're also RAF-DB's smallest
classes by far. Full breakdown, confusion matrix, ROC curves, and calibration plots are
in `notebooks/evaluation.ipynb` and `images/*_rafdb.png`.

## Project structure

```
notebooks/
  preprocessing.ipynb   # builds data/dataset_split_rafdb/{train,val,test}/<class>/ from the raw Kaggle download
  model_train.ipynb     # trains ConvNeXt-Small, saves convnext_small_rafdb.pth
  evaluation.ipynb      # metrics, confusion matrix, ROC/AUC, calibration, saved to images/
  serve.py              # FastAPI inference server
  streamlit_app.py      # GUI (upload a photo, see the predicted emotion)
images/                 # evaluation plots (checked into git)
requirements.txt
```


## Data setup

1. Download RAF-DB (aligned) from Kaggle, e.g.
   [shuvoalok/raf-db-dataset](https://www.kaggle.com/datasets/shuvoalok/raf-db-dataset)
2. Extract data.
3. Run `notebooks/preprocessing.ipynb` maps RAF-DB's numeric labels to emotion names,
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
5. **GUI** (needs the server from step 4 running in another terminal):
   ```bash
   streamlit run notebooks/streamlit_app.py
   ```
   Opens a browser page where you upload a photo and see the predicted emotion with a
   confidence chart. No command line or JSON required.

