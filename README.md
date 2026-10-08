# MLEnd Hums and Whistles

<p align="center">
  <img src="/Humming to Music Recognition.png" alt="Humming to Music Recognition" width="800"/>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/scikit--learn-1.5-orange" alt="scikit-learn">
  <img src="https://img.shields.io/badge/tensorflow-2.17-ff6f00" alt="TensorFlow">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
</p>

> Identifying which of eight songs a person is humming or whistling, from 10 seconds of audio.

---

## Overview

Query-by-humming is the task of retrieving a song from a sung, hummed or whistled fragment. It is a
useful benchmark for audio representation because a hum is a lossy and highly personal reconstruction
of a melody: performers transpose keys, drift in tempo, skip phrases, breathe audibly, and may record on
whatever device is at hand. A model that works has to be indifferent to the performer while staying
sensitive to whatever actually carries the identity of the song.

This project frames that as an eight-class classification problem and uses it to test two competing
hypotheses about where song identity lives:

- **Aggregated spectral statistics** — summarise the clip over time (average timbre, pitch-class
  content, spectral contrast) and discard the ordering of events.
- **Temporal sequence** — keep the frame-by-frame spectral trajectory and let the model learn the
  melodic contour.

The first is represented by tuned classical classifiers on engineered features, the second by an LSTM
and a 2D CNN on frame-level MFCC sequences. Both are evaluated under one protocol.

## Problem statement

Given a monophonic recording of a person humming or whistling, predict which of eight songs is being
performed.

- **Input:** one audio recording, resampled to 22.05 kHz and fixed to a 10-second window.
- **Output:** one of eight song labels.
- **Objective:** minimise misclassification on recordings from performers absent from the training
  data.
- **Metrics:** accuracy (directly interpretable against a 12.5% chance level) and macro-averaged F1
  (which refuses to reward a model that quietly abandons the harder classes).

## Objectives

1. Formalise hum/whistle song recognition as a supervised classification problem with an explicit
   quality metric and an explicit deployment assumption.
2. Establish what the dataset actually contains — class balance, participant nesting, interpretation
   styles, file integrity — before any modelling decision is taken.
3. Determine which audio representation carries song identity, by comparing nested feature sets under
   a single fixed classifier.
4. Compare classical classifiers against sequence models under identical data and identical splits.
5. Produce a generalisation estimate that is not contaminated by performer overlap or by test-set
   reuse, and quantify its uncertainty.

## Dataset

**Source:** the `MLEndHWII_Sample_800` subset of the MLEnd Hums and Whistles II dataset, a
crowdsourced collection created by students at Queen Mary University of London
([MLEnd Datasets](https://mlenddatasets.github.io/)).

**Size:** 800 `.wav` recordings across 8 song classes.

**Labels** are encoded in the filenames rather than supplied in a table:

```
S<participant>_<interpretation>_<take>_<song>.wav        e.g.  S100_hum_2_Married.wav
```

All four fields are parsed with a validated regular expression, and all four are used:

| Field | Use |
|---|---|
| `participant` | grouping variable for every split — the key methodological control |
| `interpretation` | `hum` or `whistle`; used for stratification checks and error analysis |
| `take` | repetition index; used in duplicate detection |
| `song` | the target variable |

**The dataset is not redistributed in this repository.** Obtain it from the MLEnd project and point
`DATA_DIR` at the folder containing the `.wav` files.

**Measured properties** (from the integrity audit and exploratory analysis in notebook Section 5):

| Property | Measured value |
|---|---|
| Recordings | 800, all readable, no byte-identical duplicates |
| Classes | 8, exactly 100 recordings each (perfectly balanced, chance = 12.5%) |
| Participants | 187 |
| Recordings per participant | median 4, maximum 5; **every** participant contributed more than one, so 100% of files sit in a multi-recording group |
| Interpretation style | 50% hum / 50% whistle overall, and near-even within every song |
| Duration | 10.6 s to 39.1 s (mean 21.5 s), so a 10-second window truncates most clips rather than padding them |
| Source formats | 44.1 kHz and 48 kHz, mono and stereo mixed; all loaded through `librosa` at 22.05 kHz mono |

The participant figure is the one that matters methodologically: with a median of four recordings per
performer and no performer contributing only once, a random file-level split places the same voice on
both sides of the evaluation for essentially the whole dataset.

## Methodology

### Evaluation protocol (defined before any model is fitted)

Two properties are enforced at every level of resampling, and asserted in code:

1. **Participant-disjointness.** No performer appears in both the development and test partitions,
   and none appears in both sides of any cross-validation fold. Participants contribute several
   recordings each, so a random file-level split would place the same voice on both sides; a model
   could then score well by recognising *who* is humming and exploiting the performer–song
   correlation within the dataset. That skill transfers to no new user, and it is invisible in the
   accuracy number.
2. **Class stratification**, as closely as the grouping constraint permits, so no song is absent or
   badly under-represented in any partition.

`StratifiedGroupKFold` provides both. One fold of a five-fold grouped partition becomes the test set
(≈80/20); the same splitter restricted to the development partition provides the folds used for every
feature-set choice, model choice and hyperparameter search. The test partition is read exactly once,
in Section 6.9. Preprocessing lives inside a `Pipeline`, so scalers are refitted within every fold
rather than on data the fold is about to be scored on.

### Modelling approach

**Representations**, from one extraction pass, cached under a hash of the extraction parameters:

- *Frame-level*: 20 MFCCs plus first-order deltas, as a 431 × 40 matrix per recording.
- *Aggregated*: mean and standard deviation over time of MFCCs, deltas, chroma and spectral contrast,
  organised into four nested feature sets so that the timbre-versus-tonality question can be tested
  directly.

**Models:**

| Family | Models | Inductive bias being tested |
|---|---|---|
| Classical, aggregated features | logistic regression, RBF SVM, random forest | is the song identifiable from time-averaged spectral statistics? |
| Ensemble | soft voting over the tuned SVM, tuned random forest and logistic regression | do these models fail on different recordings? |
| Sequence | LSTM (96 units, dropout) | does the ordering of frames matter? |
| Spatial | 2D CNN (3 conv blocks, batch norm, global average pooling) | are there local time–frequency motifs, wherever they occur? |

Selection proceeds representation → model family → hyperparameters, each stage on grouped
cross-validation. Networks use an inner participant-disjoint validation fold for early stopping with
best-weight restoration; their inputs are standardised using channel statistics computed on training
frames only.

### Evaluation methodology

- Computed floors, not asserted ones: majority-class and stratified-random baselines under the same
  folds.
- Accuracy and macro-F1 throughout, with cross-validation fold standard deviations.
- One test-set measurement of the selected model, with a 95% Wilson confidence interval and a
  one-sided binomial test against the 12.5% chance level.
- Per-class precision/recall/F1, plus confusion matrices in counts and row-normalised form.
- Grouped permutation importance by descriptor family, and error analysis by interpretation style and
  by song.

## Key results

Development scores are 5-fold participant-disjoint cross-validation on 640 recordings (149
performers). Test scores are a single measurement on 160 held-out recordings from 38 performers who
appear nowhere in development. Chance level is 12.5%.

### Which representation carries song identity

One fixed random forest, four nested feature sets, same folds:

| Feature set | Dimensions | CV accuracy | CV macro-F1 |
|---|---|---|---|
| A — MFCC mean (timbre only) | 20 | 0.214 ± 0.027 | 0.209 |
| B — MFCC mean + std | 40 | 0.214 ± 0.024 | 0.210 |
| C — B + deltas | 80 | 0.246 ± 0.034 | 0.247 |
| **D — C + chroma + spectral contrast** | **118** | **0.316 ± 0.051** | **0.319** |

### Model comparison

| Model | Dev accuracy | Test accuracy | Test macro-F1 |
|---|---|---|---|
| Majority-class baseline | 0.125 ± 0.022 | 0.087 | 0.020 |
| Stratified-random baseline | 0.140 ± 0.016 | — | — |
| Random forest, MFCC mean only | 0.214 ± 0.027 | — | — |
| Logistic regression, set D | 0.279 ± 0.057 | — | — |
| SVM (RBF), tuned | 0.322 | 0.331 | 0.319 |
| **Random forest, tuned — selected** | **0.326** | **0.319** | **0.301** |
| Soft-voting ensemble (SVM + RF + logistic regression) | 0.286 ± 0.056 | 0.319 | 0.302 |
| CNN (frame-level sequences) | 0.276 | 0.281 | 0.264 |
| LSTM (frame-level sequences) | 0.165 | 0.138 | 0.122 |

**Headline estimate:** the tuned random forest, selected on development data alone, classified
**51 of 160** held-out recordings correctly — **31.9% accuracy, 95% CI [24.7%, 39.7%]**, against a
12.5% chance level (one-sided binomial test, p = 1.3 × 10⁻¹⁰).

Note that the SVM scored marginally higher on the test set (33.1%) than the selected random forest.
It was **not** promoted, because selection had already been made on development data; swapping to the
test-set winner after the fact is exactly the error the protocol exists to prevent. The difference is
well inside the confidence interval either way.

## Key findings

**1. The task is learnable well above chance, but it is genuinely hard.** 31.9% accuracy on unseen
performers is roughly 2.6× chance with the interval's lower bound clearing 12.5% comfortably. It is
also nowhere near usable as a product, which is the honest reading.

**2. Pitch-class content, not timbre, carries the signal.** Adding chroma and spectral contrast lifted
cross-validated accuracy from 0.246 to 0.316, the single largest gain in the project. Doubling the
MFCC description from mean to mean-plus-standard-deviation gained nothing at all (0.214 → 0.214).
That is the expected pattern if what distinguishes songs is which notes are sung rather than the
texture of the voice singing them, and it is reassuring under a participant-disjoint protocol, since
timbre is largely a property of the performer.

Grouped permutation importance on the fitted model agrees and sharpens the point. Permuting the
chroma block cost roughly 0.16 in test accuracy, by far the largest effect, the delta block roughly
0.08, and the MFCC mean and standard-deviation blocks around 0.01 each. Spectral contrast cost almost
nothing, so the gain from feature set D is attributable to chroma specifically rather than to the two
descriptor families it added together.

**3. Sequence models did not repay their capacity at this sample size.** The CNN reached 0.276 on
validation against 0.407 on its training split, and the LSTM managed 0.165, barely above the
stratified-random floor. With ~510 recordings in the inner training split and no augmentation,
capacity is being spent memorising performers rather than learning melodic contour.

**4. The explicit ensemble did not help.** Soft voting over the tuned SVM, tuned random forest and
logistic regression scored 0.286 in cross-validation, below both tuned members individually. This is
reported rather than quietly dropped: the three members share one feature representation and their
errors are evidently correlated, which is precisely the condition under which averaging buys nothing.

**5. Hums were easier than whistles** — 39.7% against 26.1% test accuracy. This runs contrary to the
intuition that a whistle's cleaner pitch contour should be easier to classify.

**6. Per-class difficulty varies.** "Happy" reached 52.0% recall while "Married"
and "RememberMe" both sat at 16.7%, and the dominant confusions were reciprocal (Married ↔
TryEverything, Friend ↔ RememberMe), suggesting genuine melodic similarity rather than a systematic
bias toward one class.

## Limitations

- ~640 development recordings across 8 classes is a small budget for audio classification; the width
  of the reported confidence interval is the honest expression of that.
- Generalisation is estimated from a single participant-disjoint split. Repeated or nested grouped
  cross-validation would be more stable but was outside the compute budget.
- Evaluation rigour is uneven across families: classical models were selected by five-fold grouped
  cross-validation, the networks by a single inner validation fold, so cross-family comparisons are
  weaker than within-family ones.
- Only MFCC-family, chroma and spectral-contrast descriptors were tested. Explicit pitch-contour
  extraction with key and tempo normalisation, and embeddings from pre-trained audio models, were not.
- No data augmentation, so no claim is made about what the networks could reach at larger effective
  sample size.
- Labels are derived from filenames. The notebook verifies that every filename parses and that the
  class vocabulary is the expected size, but cannot verify that a contributor recorded the song the
  filename claims.
- Contributors are a self-selected crowdsourced group performing eight specific songs. Nothing here
  supports a claim about arbitrary songs or a general user population.

## Project structure

```
.
├── notebooks/
│   └── miniproject.ipynb          
├── Datasets/
│   └── MLEndHWII_Sample_800/      # not in version control; supply the .wav files here
├── artifacts/
│   ├── features_<hash>.npz        # cached features, keyed by extraction parameters
│   └── figures/                   
├── requirements.txt
├── README.md
└── LICENSE
```

## Technologies

| Purpose | Tools |
|---|---|
| Audio loading and feature extraction | `librosa`, `soundfile` |
| Classical ML, resampling, metrics | `scikit-learn` (`StratifiedGroupKFold`, `Pipeline`, `GridSearchCV`) |
| Deep learning | `tensorflow` / `keras` |
| Numerics, data handling, statistics | `numpy`, `pandas`, `scipy` |
| Visualisation | `matplotlib`, `seaborn` |
| Environment | Python 3.11+, JupyterLab |

## Reproducibility and setup

```bash
git clone <repository-url>
cd hum-to-song-recognition

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Place the audio in `Datasets/MLEndHWII_Sample_800/`, or edit `DATA_DIR` in notebook Section 5.1.

```bash
jupyter lab notebooks/miniproject.ipynb
```

Then **Kernel → Restart Kernel and Run All Cells**. The notebook is designed to run top-to-bottom
from a clean kernel and will raise an informative `FileNotFoundError` if the audio directory is
missing.

**Reproducibility measures**

- A single configuration cell holds every tunable value, so each result is traceable to an explicit
  choice.
- Seeds fixed for Python, NumPy and TensorFlow; TensorFlow deterministic op mode enabled where the
  build supports it.
- Library versions printed at the top of every run.
- Features cached under a hash of the extraction parameters, and the cache validated against the file
  list on reload, so a re-run cannot silently use a different feature matrix.
- Dependencies pinned in `requirements.txt`; no installation commands inside the notebook.
- Exact-reproduction caveat: identical numbers require the same library versions and CPU/GPU
  configuration. Small variation in the network results across hardware is expected.

Measured runtime on a 2-core cloud CPU: **14 minutes** end-to-end, of which roughly one minute is
feature extraction (cached thereafter) and the majority is network training and the grid searches.

## Future improvements

- **Melody-specific representations.** Extract a pitch contour (e.g. pYIN) and normalise for key and
  tempo, so that two performances of the same melody in different keys become comparable.
- **Augmentation.** Pitch shifting, time stretching and noise injection to multiply the effective
  number of performers, which is the direct remedy for the train–validation gap observed in the
  networks.
- **Transfer learning.** Embeddings from an audio model pre-trained on far more sound than this
  dataset contains, with a light classifier on top.
- **Nested grouped cross-validation** for all families, so that cross-family comparisons carry the
  same weight as within-family ones.
- **Ensembling across representations** rather than within one. The soft-voting ensemble of three
  classical models on the same features did not help, plausibly because their errors are correlated;
  combining a classical model with the CNN, which consumes a different representation entirely, is the
  more promising version of that idea.
- **The full ~6,000-file dataset** rather than the 800-file sample, which would test directly whether
  the sequence models were capacity-limited or data-limited.


## Acknowledgements

The MLEnd Hums and Whistles dataset was created by students and staff at the School of Electronic
Engineering and Computer Science, Queen Mary University of London, under the MLEnd Datasets
initiative.

This project was developed originally for the ECS7020P - Principles of Machine Learning (MSc Data Science and Artificial Intelligence at Queen Mary
University of London) and subsequently reworked, with
the evaluation protocol rebuilt.

## License

Code released under the MIT License. The dataset is subject to its own terms and is not redistributed
here.
