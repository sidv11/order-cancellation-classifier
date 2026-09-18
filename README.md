## 📥 Dataset

**[Online Retail II — download here (Kaggle mirror)](https://www.kaggle.com/datasets/mashlyn/online-retail-ii-uci)**

Original source / citation: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/502/online+retail+ii) — Chen, D. (2012). *Online Retail II* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5CG6D. Licensed CC BY 4.0.

> Same dataset used in the Day 2 EDA and Day 5 dashboard projects — this one asks a new question of it: can we predict which orders will get cancelled?

Place the downloaded file at `data/raw/online_retail_II.csv`.

---

# 📦 Classification: Baseline → Best — Predicting Order Cancellation

A from-scratch walkthrough of building a classifier the right way: start simple, measure honestly, catch your own mistakes, and end with a tuned model that actually improves what matters — not just what looks good on a headline number.

**The question:** using only information known at the moment an order is placed (order size, country, day/hour — never anything that happens after), can we predict whether it'll end up cancelled?

## The real story this project tells

1. **Baseline (Logistic Regression):** 86% accuracy — sounds great. Actually catches only **9%** of real cancellations. Accuracy lied.
2. **A real bug, caught and fixed:** an early version of the features scored a suspicious 100% accuracy. Turned out `Quantity` is recorded as *negative* for cancelled orders in this dataset — the raw sign was secretly leaking the answer. Fixed with `.abs()` in `features.py`.
3. **Decision Tree:** recall jumps to **80%** once the model can branch on more than one straight line.
4. **LightGBM + SMOTE (same combo as RetainIQ):** recall climbs to **87%** — the best result, with an honest tradeoff (more false alarms) clearly shown, not hidden.

## What's in this repo

```
classification-baseline-to-best/
├── data/
│   └── raw/
│       └── online_retail_II.csv
├── features.py       ← reusable order-level feature engineering (the leak fix lives here, documented)
├── notebook.ipynb      ← the full walkthrough: baseline → tree → LightGBM+SMOTE, with real numbers
├── requirements.txt
└── README.md
```

## Results, side by side

| Model | Accuracy | Cancelled Precision | Cancelled Recall | Cancelled F1 |
|---|---|---|---|---|
| Logistic Regression (baseline) | 0.859 | 0.691 | **0.087** | 0.155 |
| Decision Tree | 0.915 | 0.683 | 0.801 | 0.737 |
| **LightGBM + SMOTE** | 0.913 | 0.656 | **0.866** | **0.746** |

Accuracy barely moves across all three models — the metric that actually tells the real story is **recall on the Cancelled class**, which is what the whole project is built around.

## Why the leak mattered (and why it's left in, not hidden)

The first version of this project's features scored **100% accuracy** — a Decision Tree and LightGBM both got every single prediction right. That should never be treated as a win without investigation. It turned out `Quantity` is recorded as negative for every cancelled line item in this dataset — a direct artifact of how cancellations are logged, not a genuine risk pattern. The fix (`.abs()` on quantity/value in `features.py`) removes the sign while keeping the real signal: how big and varied the order is.

## Running it

```bash
pip install -r requirements.txt
```
Then open `notebook.ipynb` and run it top to bottom. Every cell was verified end-to-end before this was pushed.

## What I'd do next

Tune the LightGBM decision threshold deliberately based on the real business cost of a false alarm vs. a missed cancellation, rather than accepting the default 50/50 cutoff.
