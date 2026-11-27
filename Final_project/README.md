# Flirt Detection with Naive Bayes (built from scratch)

An AI Tech final project that classifies chat messages as **flirty (1)** or **not flirty (0)**, using a **Multinomial Naive Bayes classifier written from scratch in NumPy**. To check that the from-scratch model is correct, it is compared against scikit-learn's `MultinomialNB` on the same data.

The project also includes a simpler warm-up: a from-scratch Naive Bayes for **positive / negative sentiment**.

---

## Results

| Model | Task | Test accuracy |
|---|---|---|
| **Custom `MultinomialNaiveBayes`** (α = 10) | Flirt detection | **72.5%** |
| scikit-learn `MultinomialNB` (α = 10) | Flirt detection | 71.8% |
| Word-count Naive Bayes (`main.ipynb`) | Sentiment | 100% (14 / 14) |

The custom model learns the same feature counts and class priors as scikit-learn, and its accuracy is almost the same, so the from-scratch maths is right. The 100% sentiment score comes from a very small hand-written dataset, so it doesn't mean the model would do that well on real text.

---

## Notebooks

| File | What it does |
|---|---|
| [`earth.ipynb`](earth.ipynb) | **Main project:** text cleaning, class balancing, TF-IDF, then custom Naive Bayes vs scikit-learn on the flirt dataset |
| [`main.ipynb`](main.ipynb) | Warm-up: sentiment classifier with hand-written word counts, likelihoods and priors, plus an interactive "type a sentence" prediction and a confusion matrix |
| [`model.ipynb`](model.ipynb) | Early draft of the `MultinomialNaiveBayes` class |
| [`data.csv`](data.csv) | Dataset of chat messages: `label` (1 = flirty, 0 = not flirty), `text` |
| [`slang.json`](slang.json) | Slang dictionary used during cleaning (e.g. `u → you`, `idk → i do not know`) |

---

## Dataset

`data.csv` has about **5,000 messages**:

| Label | Meaning | Count |
|---|---|---|
| `0` | Not flirty | 3,421 |
| `1` | Flirty | 1,558 |

Before training, empty rows and duplicate messages are removed. The not-flirty class is then **randomly undersampled** so both classes have the same number of messages, which stops the model from just predicting the bigger class.

---

## Pipeline (`earth.ipynb`)

```
raw message
   │
   ├─ 1. lowercase + expand contractions      ("don't" → "do not")
   ├─ 2. put spaces around punctuation       (keeps "!" ";)" as their own tokens)
   ├─ 3. remove English stopwords
   ├─ 4. flag elongated words                ("heyyyy" → "hey _elongated")
   ├─ 5. replace slang + flag it             ("u" → "you _slang")
   └─ 6. lemmatize verbs                     ("matched" → "match")
   │
TF-IDF vectorizer → 80/20 train-test split → Naive Bayes
```

Steps 4 and 5 add extra marker tokens (`_elongated`, `_slang`), because stretched-out words and casual slang can be signs of flirty messages.

---

## How the from-scratch Naive Bayes works

For each class `y`, the model learns:

**Prior:** how common the class is

```
P(y) = (number of messages in class y) / (total messages)
```

**Likelihood of each word (with Laplace smoothing α):**

```
θ_yi = (N_yi + α) / (N_y + α · n_features)
```

- `N_yi`: total weight of word `i` in class `y`
- `N_y`: total weight of all words in class `y`

**Prediction:** pick the class with the highest posterior

```
P(y | x) ∝ P(y) · Π_i θ_yi ^ x_i
```

The scores are normalised so the two classes add up to 1, which gives the `predict_proba` values.

The `pipeline()` function trains both scikit-learn's model and the custom one with the same α and prints their feature counts, log priors and accuracy side by side.

---

## Getting started

```bash
pip install numpy pandas scikit-learn matplotlib seaborn nltk contractions jupyter
python -m nltk.downloader stopwords wordnet
```

Then open the main notebook and run all cells:

```bash
jupyter notebook earth.ipynb
```

`main.ipynb` asks you to type a sentence and tells you whether it's positive or negative.

---

## Possible improvements

- Use log-probabilities instead of multiplying many small numbers. That avoids the overflow / divide-by-zero warnings and fits long messages better.
- Try different α values and n-grams, or combine word counts with TF-IDF.
- Collect more flirty examples instead of throwing away not-flirty ones to balance the classes.
