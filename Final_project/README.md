# Sentiment Classifier with Naive Bayes (built from scratch)

An AI Tech final project that decides whether a sentence is **positive** or **negative**, using a **Naive Bayes classifier written by hand in Python**. Tokenizing, vocabulary building, word counting, priors and likelihoods are all implemented manually; scikit-learn is only used to split the data and report the results.

The code and the dataset are in [`main.ipynb`](main.ipynb).

---

## Results

Tested on **56 sentences the model never saw during training**:

| Metric | Positive | Negative |
|---|---|---|
| Precision | 0.71 | 0.73 |
| Recall | 0.79 | 0.57 |
| Test sentences | 28 | 28 |

**Accuracy: 67.9% (38 / 56).** 3 sentences came out as *"indescribable"* because none of their words were in the vocabulary, so both scores were equal.

The first version reported 100%, but that was because the test sentences were also used for training (see bugs fixed below).

---

## Dataset

The `data` list in the notebook has **186 sentences**: 93 positive and 93 negative.

- **46 original sentences** written for the first version. Many are deliberately tricky, mixing positive and negative words:
  - positive: *"I thought this was going to be bad but you are actually doing a wonderful job"*
  - negative: *"You might think you look good but you actually look so bad"*
- **140 added sentences** in the same style (compliments, complaints, reactions), e.g. *"Great job today, really proud of you"* and *"Such a waste of time"*.

The data is split **70 / 30**, stratified (130 train, 56 test, `random_state=42`).

---

## How it works

### 1. Tokenize

`tokenize()` turns a sentence into a list of words:

1. lowercase, and change curly apostrophes `’` to `'`
2. keep only words, so punctuation is removed (`"fail."` → `fail`)
3. remove **stopwords**: filler words like *the, is, you* that appear in both classes and only add noise
4. **negation handling**: the word after *not / no / never / …n't* gets a `not_` prefix

```
"Not bad at all, you actually look pretty cool!"
→ ['not', 'not_bad', 'actually', 'look', 'pretty', 'cool']
```

### 2. Train (training sentences only)

- `vocabuary()`: every unique word in the training set
- `word_count()`: how often each word appears in each class
- `prior_probability()`: `P(class) = sentences in class / all sentences`
- `likelyhood()`: word probabilities with Laplace smoothing (α = 1), so unseen word–class pairs don't get probability 0:

```
P(word | class) = (count of word in class + α) / (total words in class + α · vocabulary size)
```

### 3. Predict

Add up **log** probabilities (instead of multiplying many small numbers, which shrink towards 0):

```
score(class) = log P(class) + Σ log P(word | class)
```

The class with the higher score wins. Words never seen during training are skipped; if the scores are equal the result is *"indescribable"*.

---

## Bugs fixed from the first version

| Problem | Fix |
|---|---|
| Word counts were built from **all** sentences, including the test set, so the 100% accuracy was misleading | Counts now come from the training set only |
| `negative_wordcount` counted `neg_class` (original capitals) instead of `neg_vocab` (lower-cased) | Uses `neg_vocab` |
| The smoothing denominator used `+ vocabulary size` instead of `+ α · vocabulary size` | Uses the correct formula |
| Multiplying many probabilities gave tiny numbers like `5e-24` | Uses log-probabilities |
| `"fail."` and `"fail"` were different words | New `tokenize()` lower-cases words and removes punctuation |
| A leftover loop compared each word with the whole positive word list and did nothing | Removed |

---

## Getting started

```bash
pip install numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook main.ipynb
```

Run all cells. The third cell asks you to type a sentence:

```
Enter the prediction line: Major fail. You really dropped the ball on this one
Positive Score: -25.62
Negative Score: -22.90
Prediction: negative with the score of -22.90
```

---

## Ideas for further improvement

- More data: with only 186 sentences, many test words have never been seen.
- Word pairs (bigrams) such as *"not bad"* or *"waste of"* as extra features.
- Cross-validation, to get a more reliable accuracy than one 56-sentence test set.
- Tune α (`smooth_parameter`).

---

## Other files in this folder

`earth.ipynb`, `model.ipynb`, `data.csv` and `slang.json` are a friend's related project: a flirt / not-flirt message classifier that compares a from-scratch Multinomial Naive Bayes with scikit-learn's.
