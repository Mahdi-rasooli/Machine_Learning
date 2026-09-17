# Machine Learning — From Scratch and In Practice

A working notebook of my machine learning education. For every topic I study, I
write up the theory, implement the algorithm from scratch with NumPy, then check
my implementation against scikit-learn to prove it is correct.

This repository is built in public, lesson by lesson, as I progress through my
course. It is not a collection of copied tutorials — each lesson is written in my
own words, with my own implementation, my own experiments, and an honest record
of what I got wrong before I got it right.

---

## Why implement it twice?

Calling `sklearn.linear_model.LinearRegression()` proves that I can read
documentation. Deriving the cost function, differentiating it, implementing
gradient descent, and then watching my coefficients converge to the same values
scikit-learn produces proves that I understand what the library is doing.

Every lesson therefore works through the same idea twice:

| Component | Purpose |
|---|---|
| My NumPy implementation | Demonstrates the mathematics |
| scikit-learn, in the same notebook | Establishes ground truth |
| The comparison table | Shows that the two agree |

If my numbers and the library's numbers disagree, mine are wrong — and the
notebook says so.

---

## Repository structure

```
Machine_Learning/
├── README.md                  You are here
├── LICENSE
├── requirements.txt
├── docs/
│   └── repo-structure.md      Conventions and the reasoning behind them
└── lessons/
    ├── _template/             Starting point for every new lesson
    └── NN-lesson-name/
        ├── README.md          What the lesson covers, the results, reflections
        ├── notebook.ipynb     Derivation, code, and plots — the lesson itself
        ├── data/              Dataset (small and redistributable only)
        └── figures/           Plots exported for the README
```

Every lesson has the same shape, so anything you learn about reading one lesson
transfers to all the others.

---

## Roadmap

Lessons are added as I complete them in the course. The status column is honest:
nothing is marked complete until both the notebook and the write-up are finished.

| # | Lesson | Status |
|---|---|---|
| 01 | Simple Linear Regression | 🚧 In progress |
| 02 | Multiple Linear Regression | ⬜ Planned |
| 03 | Polynomial Regression and Regularization (Ridge, Lasso) | ⬜ Planned |
| 04 | Logistic Regression | ⬜ Planned |
| 05 | k-Nearest Neighbours | ⬜ Planned |
| 06 | Naive Bayes | ⬜ Planned |
| 07 | Support Vector Machines | ⬜ Planned |
| 08 | Decision Trees | ⬜ Planned |
| 09 | Ensemble Methods (Random Forest, Gradient Boosting) | ⬜ Planned |
| 10 | Model Evaluation and Cross-Validation | ⬜ Planned |
| 11 | Feature Engineering and Preprocessing Pipelines | ⬜ Planned |
| 12 | Clustering (k-Means, Hierarchical, DBSCAN) | ⬜ Planned |
| 13 | Dimensionality Reduction (PCA) | ⬜ Planned |
| 14 | Introduction to Neural Networks | ⬜ Planned |
| 15 | Capstone: an end-to-end project | ⬜ Planned |

Legend: ✅ Complete · 🚧 In progress · ⬜ Planned

---

## Running the code

```bash
git clone https://github.com/USERNAME/Machine_Learning.git
cd Machine_Learning

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Open a lesson:

```bash
jupyter notebook lessons/01-simple-linear-regression/notebook.ipynb
```

Nothing needs to be run to read the results — every notebook is committed with
its outputs intact.

---

## Conventions

- **Written in English**, so the work is readable by anyone.
- **Everything for a lesson lives in its notebook.** No helper modules to chase
  across the repository.
- **Notebook outputs are committed**, so every plot and result is visible on
  GitHub without running a single cell.
- **Implementations depend only on NumPy.** scikit-learn appears solely as a
  correctness reference.
- **Each lesson is self-contained.** No lesson depends on another.
- **Mathematics is written out**, not linked away to a video.

The reasoning behind each of these is documented in
[docs/repo-structure.md](docs/repo-structure.md).

---

## Tech stack

`Python` · `NumPy` · `pandas` · `Matplotlib` · `seaborn` · `scikit-learn` ·
`Jupyter`

---

## About

I am studying machine learning and documenting the entire process here, in the
open, including the parts I found difficult.

**Mahdi Rasooli**

- **GitHub:** [@USERNAME](https://github.com/USERNAME)
- **LinkedIn:** [Mahdi Rasooli](https://linkedin.com/in/USERNAME)

---

## License

Released under the [MIT License](LICENSE). The code is free to use. If the
write-ups help you learn something, that is the point.
