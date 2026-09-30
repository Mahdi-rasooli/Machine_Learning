# Machine Learning — A Course, Worked Through

A working notebook of my machine learning education. For every topic I study, I
write up the mathematics behind it, apply it to a real dataset, and measure how
well it does on data it has never seen.

This repository is built in public, lesson by lesson, as I progress through my
course. Each lesson states the mathematics in my own words, applies it to real
data, and reports what the model can and cannot do — including the experiments
that did not improve anything.

---

## Why the mathematics comes first

Calling `sklearn.linear_model.LinearRegression()` proves that I can read
documentation. Being able to state what that call minimises, why the answer is
exact rather than approximate, and what the resulting numbers do and do not mean
is a different thing — and it is the thing this repository is a record of.

So every lesson follows the same order:

| Step | Purpose |
|---|---|
| The problem and the intuition | What is being asked, in plain language |
| The mathematics | The model and the cost function, written out |
| The data | What was chosen, what was discarded, and why |
| The model | The library call, now unsurprising |
| The results | Measured on held-out data, then interpreted |

The last step matters most. A metric printed without interpretation says
nothing; every lesson ends by saying where its model fails and what that
failure implies.

---

## Repository structure

```
Machine_Learning/
├── README.md                  You are here
├── LICENSE
├── requirements.txt
├── docs/
│   └── repo-structure.md      Conventions and the reasoning behind them
├── lessons/
│   ├── _template/             Starting point for every new lesson
│   └── NN-lesson-name/
│       ├── README.md          What the lesson covers, and the results
│       ├── notebook.ipynb     The reasoning, code, and plots — the lesson itself
│       ├── data/              Dataset (small and redistributable only)
│       └── figures/           Plots exported for the README
└── projects/                  End-to-end work, not tied to a single technique
    └── project-name/          Same shape as a lesson
```

Every lesson has the same shape, so anything you learn about reading one lesson
transfers to all the others.

---

## Projects

Where the lessons each isolate one technique, a project applies several of them
end to end on messy real data — cleaning, exploration, feature engineering,
model comparison, tuning, and a working predictor.

| Project | What it does | Result |
|---|---|---|
| [Tehran House Price Prediction](projects/tehran-house-price/) | Predicts apartment prices from area, amenities and neighbourhood across 3,243 listings | Test R² **0.41 → 0.84**, average error halved |

---

## Roadmap

Lessons are added as I complete them in the course. The status column is honest:
nothing is marked complete until both the notebook and the write-up are finished.

| # | Lesson | Status |
|---|---|---|
| 01 | [Simple Linear Regression](lessons/01-simple-linear-regression/) | ✅ Complete |
| 02 | [Multiple Linear Regression](lessons/02-multiple-linear-regression/) | ✅ Complete |
| 03 | [Polynomial Regression](lessons/03-polynomial-regression/) | ✅ Complete |
| 04 | [Non-Linear Regression](lessons/04-non-linear-regression/) | ✅ Complete |
| 05 | [k-Nearest Neighbours](lessons/05-k-nearest-neighbours/) | ✅ Complete |
| 06 | [Decision Trees](lessons/06-decision-trees/) | ✅ Complete |
| 07 | [Logistic Regression](lessons/07-logistic-regression/) | ✅ Complete |
| 08 | [Linear vs Logistic Regression](lessons/08-linear-vs-logistic-regression/) | ✅ Complete |
| 09 | Regularization (Ridge, Lasso) | ⬜ Planned |
| 10 | Naive Bayes | ⬜ Planned |
| 11 | Support Vector Machines | ⬜ Planned |
| 12 | Ensemble Methods (Random Forest, Gradient Boosting) | ⬜ Planned |
| 13 | Model Evaluation and Cross-Validation | ⬜ Planned |
| 14 | Feature Engineering and Preprocessing Pipelines | ⬜ Planned |
| 15 | Clustering (k-Means, Hierarchical, DBSCAN) | ⬜ Planned |
| 16 | Dimensionality Reduction (PCA) | ⬜ Planned |
| 17 | Introduction to Neural Networks | ⬜ Planned |
| 18 | Capstone: an end-to-end project | ⬜ Planned |

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
- **Every model is evaluated on held-out data** wherever the dataset allows it,
  never on the rows it was fitted to. Where it does not — a short time series,
  for instance — the lesson says so and explains why.
- **Every result is interpreted**, including where the model fails.
- **Randomness is seeded**, so the numbers in the write-up are the numbers you
  get when you run it.
- **Each lesson is self-contained.** No lesson depends on another.
- **Mathematics is written out**, not linked away to a video.

The reasoning behind each of these is documented in
[docs/repo-structure.md](docs/repo-structure.md).

---

## Tech stack

`Python` · `NumPy` · `pandas` · `Matplotlib` · `seaborn` · `scikit-learn` ·
`SciPy` · `Jupyter`

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
