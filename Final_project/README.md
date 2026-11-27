# Sentiment Classifier with Naive Bayes (built from scratch)

An AI Tech final project that decides whether a sentence is **positive** or **negative**, using a **Naive Bayes classifier written by hand in plain Python**. Word counting, vocabulary building, priors and likelihoods are all implemented manually; scikit-learn is only used to split the data and report the results.

The main code is in [`main.ipynb`](main.ipynb).

---

## Example

```
Enter the prediction line: Major fail. You really dropped the ball on this one

Positive Score: 5.43e-24
Negative Score: 2.16e-23
Prediction: negative
```

---

## Dataset

A small hand-written dataset of **46 sentences** (23 positive, 23 negative), stored in the notebook. Many sentences are deliberately tricky: they mix positive and negative words, like:

| Class | Sentence |
|---|---|
| positive | *"I thought this was going to be bad but you are actually doing a wonderful job"* |
| positive | *"You are not stupid at all you are actually brilliant and amazing"* |
| negative | *"You might think you look good but you actually look so bad"* |
| negative | *"I expected a handsome guy but you are just a gross disappointment"* |

The data is split **70 / 30** into training and testing sets (`random_state=42`), which gives 14 test sentences.

---

## How it works

### 1. Build the vocabulary

Every sentence is split into words and lower-cased. `vocabuary()` collects each unique word once.

### 2. Count words per class

`word_count()` counts how many times each vocabulary word appears in the positive and in the negative sentences.

### 3. Prior probability

How common each class is (`prior_probability()`):

```
P(positive) = positive sentences / all sentences
P(negative) = negative sentences / all sentences
```

### 4. Likelihood with Laplace smoothing

How likely each word is inside a class (`likelyhood()`), with smoothing α = 1 so unseen words don't give a probability of 0:

```
P(word | class) = (count of word in class + α) / (total words in class + vocabulary size)
```

### 5. Predict

Start from the prior and multiply in the likelihood of every known word in the sentence:

```
score(class) = P(class) × Π P(word | class)
```

The class with the higher score wins. If the scores are equal, the result is *"indescribable"*.

---

## Results

| Metric | Positive | Negative |
|---|---|---|
| Precision | 1.00 | 1.00 |
| Recall | 1.00 | 1.00 |
| Test sentences | 5 | 9 |

**Accuracy: 100% (14 / 14)**, shown in a confusion matrix plotted with seaborn.

> **Note:** the word counts are built from all 46 sentences, including the 14 test sentences, so the model has already seen the test data. Counting words from `X_train` only would give a fairer accuracy.

---

## Getting started

```bash
pip install numpy scikit-learn matplotlib seaborn jupyter
jupyter notebook main.ipynb
```

Run all cells. The prediction cell asks you to type a sentence and prints the positive and negative scores and the predicted class.

---

## Possible improvements

- Build the vocabulary and word counts from the training set only (see the note above).
- `negative_wordcount` counts words from `neg_class` (original casing) instead of `neg_vocab` (lower-cased), so capitalised words like "You" are counted differently between the classes.
- Add up log-probabilities instead of multiplying, so long sentences don't produce tiny numbers that round to 0.
- Remove punctuation (e.g. `"fail."` and `"fail"` are currently different words) and collect more data.

---

## Other files in this folder

`earth.ipynb`, `model.ipynb`, `data.csv` and `slang.json` are a friend's related project: a flirt / not-flirt message classifier that compares a from-scratch Multinomial Naive Bayes with scikit-learn's.
