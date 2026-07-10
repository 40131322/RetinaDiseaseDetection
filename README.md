# Retinal Disease Classification from OCT Images

Classifying optical coherence tomography (OCT) retinal scans into four diagnostic
classes. Built for the IFT3395/IFT6390 Kaggle competition, the project works up from
linear baselines to a convolutional neural network written from scratch in NumPy, and
finally to a tuned CNN in TensorFlow that reaches ~80% test accuracy.

**Authors:** Louis-Charles Fortier · Cheng Zhang

---

## Problem

Given ~97,000 labelled 28×28 grayscale OCT images, predict one of four classes:

| Label | Diagnosis |
|:-----:|-----------|
| 0 | Choroidal neovascularization |
| 1 | Diabetic macular edema |
| 2 | Drusen |
| 3 | Healthy retina |

The test set is 1,000 unlabelled images and the evaluation metric is classification
accuracy.

## Results

| Model | Public | Private |
|-------|:------:|:-------:|
| Logistic regression | baseline | baseline |
| CNN — NumPy from scratch (15 epochs) | 56.5% | 57.3% |
| CNN — TensorFlow, tuned + bootstrapped voting | **75.5%** | **80.1%** |

Highlights from the analysis:

- A two-convolution CNN with kernel widths of 7 and 5 and moderate capacity gave the
  best trade-off between accuracy and training time.
- Gaussian filtering produced no significant gain; Local Binary Pattern features
  slightly *hurt* performance and encouraged earlier overfitting.
- Bootstrapped ensemble voting reduced variance for a ~2% lift on both leaderboards.
- The confusion matrix shows most remaining error is healthy retinas (class 3) being
  predicted as drusen (class 2) — the clearest target for future improvement.

The full write-up with figures and tables is in [`report/`](report/).

## Repository structure

```
.
├── notebooks/
│   ├── 01_linear_baselines.ipynb    # logistic regression baseline
│   ├── 02_cnn_numpy.ipynb           # from-scratch NumPy CNN   (56.5% / 57.3%)
│   ├── 03_cnn_kernel_search.ipynb   # kernel-width hyperparameter sweeps
│   └── 04_cnn_final.ipynb           # final TensorFlow CNN      (75.5% / 80.1%)
├── src/
│   ├── functions.py                 # shared preprocessing / helper functions
│   └── perceptron.py                # perceptron implementation
├── results/
│   └── submission.csv               # best Kaggle submission
├── report/
│   └── report.pdf                   # full report with figures and tables
├── old/                             # earlier iterations, kept for reference
├── requirements.txt
└── README.md
```

`old/` holds development notebooks, alternate models, and superseded submissions. It is
not needed to reproduce the results and can be ignored.

## Getting started

```bash
pip install -r requirements.txt
jupyter lab
```

The dataset is not included in this repository. Download `train_data.pkl` and
`test_data.pkl` from the
[Kaggle competition page](https://www.kaggle.com/competitions/ift3395-ift6390-identification-maladies-retine)
and place them in a `data/` folder at the project root. Then run the notebooks in
numeric order — each is self-contained.

## Methods

**Preprocessing.** The classes are imbalanced (0: 27k, 1: 8k, 2: 6k, 3: 37k). Two
strategies address this: undersampling every class down to the smallest, and a
bootstrapping scheme that rebuilds several balanced training sets from the full data.
Optional Gaussian filtering and LBP feature extraction were also evaluated.

**Baseline.** Logistic regression, used as the reference point the CNNs must beat.

**NumPy CNN.** A single conv → ReLU → max-pool → dense → logits network implemented
without a deep-learning framework, to demonstrate the mechanics end to end.

**TensorFlow CNN.** Two conv/ReLU/max-pool blocks with dropout (0.25), a dense layer,
and a softmax output. Kernel widths, capacity, and input representation were tuned on a
20% validation split; the final model is an ensemble combined by bootstrap voting.

## Possible improvements

L1/L2 regularization with a smaller learning rate, stronger domain-informed feature
engineering aimed at the class 2/3 confusion, and augmentation robustness testing
(small rotations and mirroring) are the most promising next steps.
