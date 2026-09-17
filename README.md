# Machine Learning — A Course, Worked Through

A working notebook of my machine learning education. For every topic I study, I
write up the mathematics behind it, apply it to a real dataset, and measure how
well it does on data it has never seen.

This repository is built in public, lesson by lesson, as I progress through my
course. It is not a collection of copied tutorials — each lesson is written in my
own words, with my own implementation, my own experiments, and an honest record
of what I got wrong before I got it right.

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
| 01 | [Simple Linear Regression](lessons/01-simple-linear-regression/) | ✅ Complete |
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
- **Every model is evaluated on held-out data**, never on the rows it was
  fitted to.
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
