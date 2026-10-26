# Machine Learning — A Course, Worked Through

A machine learning portfolio: **16 lessons** across supervised learning,
unsupervised learning and recommender systems, and **4 end-to-end projects**.
For every topic the mathematics is written out, applied to a real dataset, and
measured on data the model has never seen — and every result is interpreted,
including where the model fails.

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

## Projects

Where the lessons each isolate one technique, a project applies several of them
end to end on real data — cleaning, exploration, feature engineering, model
comparison, tuning, and a working result.

| Project | What it does | Result |
|---|---|---|
| [Tehran House Price Prediction](projects/tehran-house-price/) | Predicts apartment prices from area, amenities and neighbourhood across 3,243 listings | Test R² **0.41 → 0.84**, average error halved |
| [Heart Attack Prediction](projects/heart-attack-prediction/) | Classifies 302 patients from 13 clinical measurements, comparing KNN, logistic regression, a decision tree and SVMs | Test accuracy **0.84** against a 0.54 baseline |
| [Mall Customer Segmentation](projects/mall-customer-segmentation/) | Groups 200 shoppers by income and spending, comparing k-means, hierarchical clustering and DBSCAN | **5 segments**, silhouette **0.55**, k-means and hierarchical agreeing on 196 of 200 customers |
| [Movie Recommender System](projects/movie-recommender/) | Recommends movies from 3.9 million MovieLens ratings, comparing content-based and collaborative filtering for the same user | The two top-10 lists share **no movies**: genre matches against what similar viewers loved |

---

## Roadmap

The lessons, grouped by area. Lesson numbers record the order in which the
lessons were completed, so within the groups below they are not always
consecutive.

### 1. Supervised learning

Learning from examples that come with the right answer.

**Regression** — predicting a number

| # | Lesson | What it covers |
|---|---|---|
| 01 | [Simple Linear Regression](lessons/01-simple-linear-regression/) | Fitting a straight line through data to predict a continuous value |
| 02 | [Multiple Linear Regression](lessons/02-multiple-linear-regression/) | Several input features at once — fitting a plane instead of a line |
| 03 | [Polynomial Regression](lessons/03-polynomial-regression/) | Fitting a curve while still using linear regression to do it |
| 04 | [Non-Linear Regression](lessons/04-non-linear-regression/) | A model whose shape is chosen in advance, when no line or polynomial will do |

**Classification** — predicting a category

| # | Lesson | What it covers |
|---|---|---|
| 05 | [k-Nearest Neighbours](lessons/05-k-nearest-neighbours/) | Classifying a new case by letting the most similar known cases vote |
| 06 | [Decision Trees](lessons/06-decision-trees/) | A short sequence of yes/no questions, each chosen to leave the groups as pure as possible |
| 07 | [Logistic Regression](lessons/07-logistic-regression/) | Predicting the probability of a yes/no outcome, learned by gradient descent |
| 08 | [Linear vs Logistic Regression](lessons/08-linear-vs-logistic-regression/) | The same weighted sum asked two different questions — and how to choose |
| 11 | [Support Vector Machines](lessons/11-support-vector-machines/) | The widest possible gap between classes, bent with a kernel when a straight one will not do |

**Ensemble methods** — combining many models into a stronger one

| # | Lesson | What it covers |
|---|---|---|
| 09 | [Random Forest](lessons/09-random-forest/) | Many deliberately different trees voting together, and beating any one of them |
| 10 | [XGBoost](lessons/10-xgboost/) | Hundreds of weak trees built in sequence, each correcting the mistakes of the last |

Applied in: [Tehran House Price Prediction](projects/tehran-house-price/) ·
[Heart Attack Prediction](projects/heart-attack-prediction/)

### 2. Unsupervised learning

Finding structure in data that has no answers attached.

**Clustering**

| # | Lesson | What it covers |
|---|---|---|
| 12 | [K-Means Clustering](lessons/12-k-means-clustering/) | Grouping unlabelled data around centroids — and checking what the groups really mean |
| 13 | [Hierarchical Clustering](lessons/13-hierarchical-clustering/) | A whole family tree of groups, cut afterwards at the level that fits the question |
| 14 | [DBSCAN](lessons/14-dbscan/) | Clusters as dense regions of any shape, with an honest label for points that belong nowhere; ends with HDBSCAN |

Applied in: [Mall Customer Segmentation](projects/mall-customer-segmentation/)

### 3. Recommender systems

Predicting what a user will like next.

| # | Lesson | What it covers |
|---|---|---|
| 15 | [Content-Based Filtering](lessons/15-content-based-recommender/) | Learning what a user likes about the movies they rated, and recommending more with those qualities |
| 16 | [Collaborative Filtering](lessons/16-collaborative-filtering/) | Recommending what people with similar taste enjoyed, without looking at the movies themselves |

Applied in: [Movie Recommender System](projects/movie-recommender/)

### Next

Regularization (Ridge and Lasso), Naive Bayes, model evaluation and
cross-validation, dimensionality reduction (PCA), and an introduction to neural
networks.

---

## Repository structure

```
Machine_Learning/
├── README.md                  You are here
├── LICENSE
├── requirements.txt
├── docs/
│   └── repo-structure.md      Conventions and the reasoning behind them
├── lessons/                   16 lessons, one technique each
│   └── NN-lesson-name/
│       ├── README.md          Summary: the mathematics, the data, the results
│       ├── notebook.ipynb     The full working — reasoning, code and plots
│       ├── data/              The dataset
│       └── figures/           Plots shown in the README
└── projects/                  4 end-to-end projects, in the same shape
```

Every lesson and project has the same shape, so anything learned about reading
one transfers to all the others.

---

## Running the code

```bash
git clone https://github.com/Mahdi-rasooli/Machine_Learning.git
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
- **Randomness is seeded**, so the numbers in the write-ups are the numbers you
  get when you run the notebooks.
- **Each lesson stands on its own** and can be read and run without the others.
  The one shared resource is the MovieLens data, stored once in lesson 15 and
  read from there by lesson 16 and the movie recommender project.
- **Mathematics is written out**, not linked away to a video.
- **Sources are credited** — the course, the datasets and the papers behind
  each method.

The reasoning behind each of these is documented in
[docs/repo-structure.md](docs/repo-structure.md).

---

## Tech stack

`Python` · `NumPy` · `pandas` · `Matplotlib` · `seaborn` · `scikit-learn` ·
`XGBoost` · `Cartopy` · `Jupyter`

---

## About

This repository is my machine learning portfolio. The lessons follow the topics
of the *Machine Learning with Python* course by Jadi, which is credited in each
lesson; the four projects apply those methods end to end to problems of their
own, from Tehran house prices to movie recommendations.

Every notebook runs from top to bottom with seeded randomness, and every number
quoted in a README comes from those runs.

**Mahdi Rasooli** · [GitHub @Mahdi-rasooli](https://github.com/Mahdi-rasooli)

---

## License

Released under the [MIT License](LICENSE). The code is free to use. If the
write-ups help you learn something, that is the point.
