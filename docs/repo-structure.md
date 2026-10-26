# Repository structure and conventions

This document records how the repository is organized and, more importantly, why.
Every convention below was a deliberate decision, and each one has a cost that
was weighed before it was adopted.

---

## One shape for every lesson

```
lessons/NN-lesson-name/
├── README.md         What the lesson covers, and the results
├── notebook.ipynb    The reasoning, code, and plots — the lesson itself
├── data/             Dataset, if small and redistributable
└── figures/          Plots exported for README.md
```

Uniformity is the point. A reader who understands how to navigate one lesson can
navigate all of them, and adding a lesson becomes a mechanical act rather than a
design decision made fifteen separate times.

Lessons are numbered with a two-digit prefix (`01-`, `02-`) so that directory
listings stay in course order rather than alphabetical order.

**Lessons stand on their own.** When a lesson builds on an earlier technique —
the random forest on the decision tree, for instance — it uses scikit-learn's
version or builds what it needs itself; it never imports code from another
lesson's folder. Self-contained lessons can be read and run in isolation, which
is how anyone evaluating this repository will actually encounter them.

The one deliberate exception is data, described under *Data policy* below.

---

## Everything for a lesson lives in its notebook

There are no `src/` modules and no helper packages. The mathematics, the data
preparation, the model, and the plots all sit in a single `notebook.ipynb`, in
the order a reader would want to meet them.

The usual argument for extracting code into modules is reuse and testability. It
does not apply here: nothing is reused between lessons, and each piece of code
exists to be *read* alongside the explanation that motivates it. Moving the
model-fitting code into a separate file would put the mathematics on one screen
and the code applying it on another, which defeats the purpose.

The cost is real — notebooks produce noisier diffs than `.py` files, and there is
no automated test suite. Both were accepted knowingly in exchange for lessons
that read as a single continuous argument.

---

## README and notebook divide the work

The notebook is the working artifact: full derivation, every cell, every
experiment, including the ones that did not pan out.

The lesson README is the summary: what the lesson covers, the mathematics in
condensed form, the dataset, the final numbers, and what was learned. It is
written for someone browsing GitHub who has thirty seconds, not thirty minutes.

Some overlap between the two is intended. The README is not an index of the
notebook — it is a shorter answer to the same question.

---

## The mathematics comes before the library call

Each notebook states the model and its cost function before any estimator is
constructed. By the time `fit()` appears, the reader already knows what quantity
is being minimised and why the result takes the form it does.

Using a library proves familiarity with an API. Being able to say what the
library is computing, why that computation has the properties it has, and what
its output means for the data at hand is the claim this repository exists to
support.

This is also why no lesson ends at a metric. A number without an interpretation
demonstrates that code ran, not that anything was understood, so every lesson
closes by saying where its model breaks down and what that implies about the
problem.

---

## Notebook outputs are committed

Standard practice is to strip notebook outputs before committing, because stored
outputs produce large, noisy diffs.

This repository does the opposite, deliberately. Its primary audience reads it in
a browser on GitHub and will not clone it, install dependencies, and execute
cells. If outputs are stripped, that reader sees code and no results — no
regression line, no loss curve, no confusion matrix. Since the notebook *is* the
lesson here, stripping outputs would empty out the lesson.

Practically: never run "Clear All Outputs" before committing.

---

## Data policy

Datasets are committed under each lesson's or project's `data/` folder, so
everything runs immediately after cloning. Each README states where its data
comes from and credits it.

Two rules keep the repository small:

- **Nothing over GitHub's 100 MB file limit.** The MovieLens ratings file is
  102 MB as a CSV, so it is committed gzip-compressed (`.csv.gz`, 26 MB); pandas
  reads compressed CSV files directly, so no code has to change.
- **A large dataset is stored once.** Lessons 15 and 16 and the movie
  recommender project all use the same MovieLens files. They live in lesson 15's
  `data/` folder, and the other two read them through a relative path instead of
  keeping copies. Small datasets — a few kilobytes — are simply copied, so that
  each lesson keeps its own.

Anything restrictively licensed is not committed at all. A reader must never be
blocked by a missing file with no explanation of where it came from.

---

## Everything is written in English

The audience for this repository includes people who do not share my first
language. Writing in English is what makes the work legible to them.

---

## Mathematics is written out

Each lesson states the model, its cost function, and the update rule in LaTeX,
which GitHub renders natively in both notebooks and Markdown. Linking to a
lecture video instead would delegate the explanation to someone else; the ability
to state the mathematics plainly is part of what is being demonstrated.

---

## Every lesson credits its sources

Each lesson ends with references: the course lesson it follows, where the
dataset came from, and any documentation that was actually used.

Lessons based on a course are not presented as though they were invented here.
Saying plainly what was followed and what was worked out is what makes the rest
of the write-up trustworthy.

---

## Lessons and projects are different things

A **lesson** isolates one technique and asks what it can do. The dataset is
chosen to make the method legible, and the write-up is about the method.

A **project** in `projects/` starts from a question rather than a technique, and
whatever it takes to answer it is fair game: cleaning a broken column,
engineering features, comparing several models, tuning a hyperparameter,
shipping something that makes predictions. The write-up is about the problem.

Projects use the same folder shape as lessons — `README.md`, `notebook.ipynb`,
`data/`, `figures/` — so nothing new has to be learned to read one. They are
listed above the lesson roadmap in the root README because they show more.

---

## Material I did not write stays out

Course handouts, instructor notes and any other text written by someone else are
listed in `.gitignore` (`notes/`, `*.docx`) and never committed, even when they
are useful and even when they describe my own work.

Anything from them worth keeping gets rewritten in my own words and credited in
that lesson's or project's references. A repository presented as my own work has
to contain only my own work.

---

## Adding a new lesson

1. Create `lessons/NN-lesson-name/` with the next free number, in the shape
   shown at the top of this document.
2. Write the notebook in the standard order — the problem, the intuition, the
   mathematics, the data, the model, the results, the references — and make
   sure its final plot is saved into `figures/` so the README can display it.
3. Execute it from top to bottom before committing, so every committed output
   comes from one clean run.
4. Condense it into the README, using any existing lesson as the model:
   *What this lesson covers*, *Intuition*, *The mathematics*, *The dataset*,
   *What the code does*, *Results*, *References*.
5. Add the lesson to the right group of the roadmap in the root `README.md` —
   the roadmap is only useful while it is accurate.
