# Trump vs. Staffer Tweet Classification

This project aims to classify tweets from the `@realDonaldTrump` Twitter account as either authored by Donald Trump himself or by his staff. The task focuses on distinguishing subtle stylistic and temporal differences in tweet content, device usage, and linguistic structure using both classical machine learning and transformer-based models.

---

## 🧹 Dataset Filtering and Assumptions

- **Account Selection**: Only tweets from the `@realDonaldTrump` handle were included (3,144 entries) to focus on personal versus staff-authored tweets.
- **Device-Based Filtering**: Tweets sent via **Android** were assumed to be authored by Trump, while **iPhone** tweets were assumed to be written by staffers.
- After filtering to only Android and iPhone tweets, the final dataset contained **2,895 tweets**:
  - **Android**: 1,991 (Trump-authored → label `0`)
  - **iPhone**: 904 (Staff-authored → label `1`)

---

## 🔧 Feature Engineering

Several features were extracted to enrich the model beyond raw text:

- **Tweet length**
- **Tweet time**: Hour extracted and binned into `morning`, `afternoon`, `evening`, `night`
- **Year**: Categorical representation
- **Capitalization patterns**: 
  - `full_caps_count`: ALL CAPS words
  - `init_caps_count`: Words starting with capital letters
- **Non-letter character count**: Emojis, punctuation, links
- **Part-of-speech counts**: Nouns, verbs, adjectives, adverbs, pronouns

---

## ⚙️ Preprocessing Pipeline

- **Text**: Vectorized using `TfidfVectorizer` (max 5000 features, English stop words removed)
- **Categorical**: One-hot encoded (e.g., tweet time, year)
- **Numeric**: Standardized using `StandardScaler`

All classifiers used the same processed 13-feature representation.

---

## 🤖 Models Used

### Classical Models
- **Logistic Regression**
- **Linear SVM**
- **Non-Linear SVM**
- **Feedforward Neural Network (FFNN)**:
  - Architecture: 256 → 128 → 64 → 32 hidden layers
  - Dropout: 30%, 20%
  - Optimizer: Adam
  - Loss: Binary Crossentropy
- **XGBoost**: Selected for its performance with structured data and interpretability

### Transformer-Based Model
- **BERT (bert-base-uncased)** via HuggingFace Transformers
  - Used only tweet text and label
  - Fine-tuned for 3 epochs with batch size of 16
  - Tokenization with padding and truncation
  - Evaluated with `Trainer` API

---

## 📊 Evaluation Metrics

- **Accuracy** – Overall performance
- **AUC (ROC)** – Class separation power
- **F1-score (Staffer)** – Performance on minority class
- **Macro F1** – Balanced F1 across both classes

---

## 🏆 Results Summary

| Model               | Accuracy | AUC    | F1 (Staffer) | Macro F1 |
|--------------------|----------|--------|--------------|----------|
| Logistic Regression| 87.05%   | 0.9286 | 0.77         | 0.77     |
| Linear SVM         | 87.39%   | 0.9374 | 0.78         | 0.78     |
| Non-Linear SVM     | 85.00%   | 0.9117 | 0.76         | 0.76     |
| FFNN               | 87.56%   | 0.9233 | 0.84         | 0.84     |
| **XGBoost**        | 89.12%   | 0.9445 | 0.81         | 0.85     |
| **Transformer (BERT)** | **92.22%** | **0.9636** | **0.87** | **0.91** |

---

## 🧠 Analysis & Insights

- **BERT** outperformed all models, showing its strength in capturing nuanced language differences without needing manual features.
- **XGBoost** led the classical models, benefiting from its handling of complex, structured data.
- Classical models like SVM, FFNN, and Logistic Regression performed well with engineered features but lacked deep language understanding.
- **Non-Linear SVM** underperformed, likely due to sensitivity to high-dimensional TF-IDF features.

---

## 🗣 Stylometric Insights

- **Trump’s tweets** (Android): More spontaneous and expressive, using ALL CAPS, frequent adjectives, and consistent posting times (e.g., noon).
- **Staff tweets** (iPhone): More structured, professional, and promotional; high frequency of links (`https`), hashtags, and formal campaign language.

---

## 📌 Conclusion

Transformer-based models like BERT significantly improve performance in text classification tasks involving subtle stylistic differences. However, combining structured linguistic features with strong tabular models (like XGBoost) still yields competitive results — especially when deep models are impractical.

---

## 🧾 Acknowledgments

- Pre-trained BERT models provided by HuggingFace 🤗

---

