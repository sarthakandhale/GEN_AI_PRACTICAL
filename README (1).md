# genai_prac2
# BBC News Topic Classification using an Embedding + LSTM Network

A deep-learning NLP assignment that classifies BBC news articles into one of **5 topic categories** (business, entertainment, politics, sport, tech) using an `Embedding` layer feeding a `Bidirectional LSTM`, benchmarked against a classical TF-IDF + Logistic Regression baseline.

---

## 1. Dataset Information

| Property | Value |
|---|---|
| Name | BBC News dataset |
| Source | Greene & Cunningham (2006), BBC news articles 2004-2005 |
| Total articles (after cleaning) | 2,127 |
| Number of classes | 5 (business, entertainment, politics, sport, tech) |
| business | 503 (~23.7%) |
| entertainment | 369 (~17.3%) |
| politics | 403 (~19.0%) |
| sport | 505 (~23.7%) |
| tech | 347 (~16.3%) |
| Columns | `label` (category name), `text` (article title + body) |
| Class balance | Fairly balanced (~16-24% per class) |
| Avg. article length | ~384 words (min 89, max 4,432) |

The notebook loads the data from a local `bbc_news_dataset.csv`, and automatically falls back to rebuilding it from the original tab-separated BBC news file hosted on GitHub if the file is missing — making it fully self-contained and reproducible. A stratified 80/20 train-test split was used (Train: 1,701 articles, Test: 426 articles).

---

## 2. Concepts Used

- **Text preprocessing:** lowercasing, URL normalization, punctuation stripping, whitespace collapsing
- **Tokenization & sequence padding:** Keras `Tokenizer` (vocabulary capped at 20,000 words, `<OOV>` token for rare words) and `pad_sequences` (fixed length of 400 tokens, post-padding/truncating — chosen to cover the bulk of article lengths without excessive padding)
- **Word Embeddings:** a trainable `Embedding` layer (100-dimensional) learned from scratch
- **Recurrent Neural Network:** `Bidirectional LSTM` (64 units) to capture context from both directions of an article
- **Regularization:** `SpatialDropout1D` and `Dropout` layers to prevent overfitting
- **Multi-class handling:** integer labels via `LabelEncoder`, `softmax` output layer, `sparse_categorical_crossentropy` loss, and computed `class_weight` (balanced) passed into training
- **Classical ML baseline:** TF-IDF vectorization (unigrams + bigrams, 20,000 features) + multinomial Logistic Regression, used as a sanity-check benchmark
- **Training strategy:** stratified train/test split (80/20), validation split, `EarlyStopping` (patience = 3, restores best weights)
- **Evaluation metrics:** Accuracy, per-class Precision/Recall/F1, macro-averaged metrics, 5x5 Confusion Matrix, one-vs-rest ROC-AUC and ROC curves
- **Exploratory Data Analysis:** class distribution plots, article-length distributions, and **word clouds** per category
- **Error analysis:** inspection of misclassified articles and most-confused category pairs

---

## 3. Word Cloud

Word clouds were generated separately for each of the **5 categories** (using the `wordcloud` library) to visually surface the most frequent terms in each topic.

- **Business:** dominated by terms such as *"market"*, *"company"*, *"growth"*, *"shares"*, *"profit"*, *"sales"*, *"economy"*.
- **Entertainment:** dominated by terms such as *"film"*, *"best"*, *"actor"*, *"award"*, *"show"*, *"music"*, *"star"*.
- **Politics:** dominated by terms such as *"government"*, *"labour"*, *"election"*, *"minister"*, *"party"*, *"blair"*.
- **Sport:** dominated by terms such as *"game"*, *"player"*, *"win"*, *"team"*, *"match"*, *"club"*.
- **Tech:** dominated by terms such as *"technology"*, *"user"*, *"mobile"*, *"net"*, *"digital"*, *"software"*.

This visual contrast confirms that vocabulary alone carries a strong, learnable signal for separating the five topics, which both the baseline and the LSTM exploit.

---

## 4. Model Architecture

```
Embedding(vocab_size=20000, output_dim=100, input_length=400)
SpatialDropout1D(0.2)
Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
Dense(32, activation="relu")
Dropout(0.3)
Dense(5, activation="softmax")
```

- **Total parameters:** 2,088,773
- **Loss:** Sparse Categorical Crossentropy
- **Optimizer:** Adam (lr = 1e-3)
- **Training:** 9 epochs (early-stopped from a max of 15, restores best weights), batch size 32, class-weighted

---

## 5. Results

### Baseline — TF-IDF + Logistic Regression
| Category | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| business | 1.00 | 0.97 | 0.98 | 101 |
| entertainment | 0.99 | 1.00 | 0.99 | 74 |
| politics | 0.98 | 0.98 | 0.98 | 81 |
| sport | 1.00 | 1.00 | 1.00 | 101 |
| tech | 0.97 | 1.00 | 0.99 | 69 |

Overall accuracy: **0.99**  |  Macro ROC-AUC (one-vs-rest): **0.9971**

### Deep Model — Embedding + Bidirectional LSTM
| Category | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| business | 0.9238 | 0.9604 | 0.9417 | 101 |
| entertainment | 0.8333 | 0.9459 | 0.8861 | 74 |
| politics | 0.9863 | 0.8889 | 0.9351 | 81 |
| sport | 0.9892 | 0.9109 | 0.9485 | 101 |
| tech | 0.9014 | 0.9275 | 0.9143 | 69 |

Overall accuracy: **0.9272**  |  Macro Precision: **0.9268**  |  Macro Recall: **0.9267**  |  Macro F1: **0.9251**  |  Macro ROC-AUC (one-vs-rest): **0.9903**

**Confusion Matrix (LSTM, test set of 426 articles):**

| Actual \ Predicted | business | entertainment | politics | sport | tech |
|---|---|---|---|---|---|
| **business** | 97 | 1 | 0 | 0 | 3 |
| **entertainment** | 1 | 70 | 1 | 0 | 2 |
| **politics** | 2 | 4 | 72 | 1 | 2 |
| **sport** | 0 | 9 | 0 | 92 | 0 |
| **tech** | 5 | 0 | 0 | 0 | 64 |

Total misclassified: 31 out of 426 test articles (7.28%). The most common confusion pairs were **sport → entertainment** (9 cases) and **tech → business** (5 cases).

### Baseline vs. LSTM — Side by Side
| Model | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | ROC-AUC (macro, ovr) |
|---|---|---|---|---|---|
| TF-IDF + Logistic Regression | 0.99 | 0.99 | 0.99 | 0.99 | 0.9971 |
| Embedding + Bidirectional LSTM | 0.9272 | 0.9268 | 0.9267 | 0.9251 | 0.9903 |

**Key takeaway:** both models perform strongly, since the five BBC News topics use fairly distinct vocabulary. The TF-IDF baseline is competitive with — and here, ahead of — the LSTM, a reminder that deep sequence models typically need more data to clearly outperform strong classical baselines on a dataset of only ~2,100 documents. Because the task is multi-class, per-class precision/recall/F1 and the confusion matrix are far more informative than a single accuracy number, since they reveal exactly which topic pairs get confused.

### Custom Predictions (sample)
| Text (truncated) | Prediction | Confidence |
|---|---|---|
| "The stock market rallied today after the central bank announced..." | business | 0.9942 |
| "The actress accepted the award for best performance at last night's ceremony..." | entertainment | 0.9797 |
| "The prime minister addressed parliament today, defending the government's..." | politics | 0.6249 |
| "The home team clinched the championship title after a dramatic final match..." | entertainment* | 0.7022 |
| "The company unveiled its latest smartphone, featuring a faster processor..." | business* | 0.6709 |

\*Two of the five hand-written samples were misclassified (a sport article predicted as entertainment, and a tech-product article predicted as business), both with comparatively low confidence — consistent with the sport/entertainment and tech/business confusion already visible in the confusion matrix above.

---

## 6. Possible Extensions
- Use pre-trained embeddings (GloVe/Word2Vec) instead of learning from scratch
- Try a deeper/stacked LSTM or a GRU
- Add attention over LSTM outputs
- Tune `MAX_LEN`, `VOCAB_SIZE`, or use validation-set sweeps to reduce the sport/entertainment and tech/business confusion

---

## 7. How to Run
1. Open `BBC_News_Classification_Embedding_LSTM.ipynb` in Jupyter.
2. Run all cells top to bottom — the dataset auto-downloads/rebuilds if not present locally.
3. Requires: `tensorflow`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `wordcloud`.
