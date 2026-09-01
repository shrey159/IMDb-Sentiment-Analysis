# IMDb Sentiment Analysis with TF-IDF and χ² Feature Selection

## Overview

This project explores sentiment classification of IMDb movie reviews using
traditional machine learning and TF-IDF text representation.

The main focus of the project is not only achieving high classification
accuracy, but also investigating whether the high-dimensional TF-IDF feature
space can be significantly reduced using χ² (Chi-Square) feature selection
without substantially affecting model performance.

## Objective

The project aims to answer the following question:

> How much can the TF-IDF feature space be reduced using χ² feature selection
> while preserving sentiment classification performance?

The project includes:

- Text preprocessing
- TF-IDF vectorization
- Baseline model comparison
- χ² feature selection
- Evaluation across different feature counts
- Confusion matrix analysis
- End-to-end prediction on an unseen review

## Dataset

The project uses the IMDb Movie Review dataset.

- 50,000 labeled reviews
- 25,000 training reviews
- 25,000 test reviews
- Positive sentiment → `1`
- Negative sentiment → `0`

The training dataset initially contained 96 duplicate reviews. These were
removed before model training, leaving 24,904 unique training reviews.

## Methodology

The overall pipeline is:

Raw Review
↓
Text Preprocessing
↓
TF-IDF Vectorization
↓
χ² Feature Selection
↓
Logistic Regression
↓
Sentiment Prediction

### 1. Text Preprocessing

The reviews are cleaned before feature extraction. The preprocessing includes:

- Converting text to lowercase
- Removing HTML tags
- Removing unnecessary punctuation and special characters
- Normalizing whitespace

### 2. TF-IDF Vectorization

TF-IDF (Term Frequency-Inverse Document Frequency) is used to convert
the cleaned reviews into numerical feature vectors.

The vectorizer is fitted only on the training data and subsequently used to
transform the test data to avoid data leakage.

The resulting representation contains 74,849 features.

### 3. Baseline Models

Three traditional machine learning classifiers were evaluated using the
full TF-IDF representation:

- Logistic Regression
- Naive Bayes
- Linear SVM

### 4. χ² Feature Selection

χ² feature selection was used to identify features that have a strong
statistical association with the sentiment labels.

Different values of `K` were evaluated:

- 1,000
- 2,000
- 5,000
- 10,000
- 20,000
- 30,000
- 40,000

Here, `K` represents the number of features retained after feature selection.

## Results

### Baseline Model Comparison

| Model | Features | Accuracy |
|---|---:|---:|
| Logistic Regression | 74,849 | 88.29% |
| Naive Bayes | 74,849 | 83.14% |
| Linear SVM | 74,849 | 87.72% |

Logistic Regression achieved the strongest baseline performance.

### χ² Feature Selection

| K | Accuracy | F1 Score |
|---:|---:|---:|
| 1,000 | 86.53% | 86.68% |
| 2,000 | 87.32% | 87.40% |
| 5,000 | 87.91% | 87.96% |
| 10,000 | 87.98% | 88.04% |
| 20,000 | 88.21% | 88.25% |
| 30,000 | 88.20% | 88.23% |
| 40,000 | 88.21% | 88.24% |
| 74,849 | 88.29% | 88.29% |

The results show that performance begins to plateau around 20,000 features.

### Final Model

The final selected model uses:

- TF-IDF
- χ² feature selection
- 20,000 selected features
- Logistic Regression

Performance:

- Original features: **74,849**
- Selected features: **20,000**
- Feature reduction: **73.28%**
- Full-model accuracy: **88.29%**
- χ² model accuracy: **88.21%**
- Full-model F1: **88.29%**
- χ² model F1: **88.25%**

The selected model retains approximately **99.91% of the baseline accuracy**
while using only about **26.72% of the original features**.

## Confusion Matrix

The final χ² + Logistic Regression model produced:

| | Predicted Negative | Predicted Positive |
|---|---:|---:|
| Actual Negative | 10,990 | 1,510 |
| Actual Positive | 1,437 | 11,063 |

The errors are relatively balanced between the two sentiment classes,
indicating that the model does not strongly favor one class.

## Example Prediction

The trained pipeline was tested on a previously unseen movie review.

Example:

> This movie was absolutely fantastic. The story was engaging, the acting was
> excellent, and I really enjoyed every minute of it.

Prediction:

**Positive**

## Key Findings

1. Logistic Regression provided the strongest baseline performance.
2. χ² feature selection reduced the TF-IDF feature space by **73.28%**.
3. Reducing the feature space from 74,849 to 20,000 features resulted in only
   a **0.08 percentage-point decrease in accuracy**.
4. Model performance largely plateaued after approximately 20,000 features.
5. The results demonstrate that many of the original TF-IDF features provide
   limited additional predictive value.

## Limitations

- The approach uses traditional machine learning rather than transformer-based
  language models.
- The preprocessing pipeline does not explicitly model complex linguistic
  phenomena such as sarcasm or context.
- Mutual Information was explored as an alternative feature-selection method,
  but its computational cost on the high-dimensional TF-IDF representation
  made it impractical for the final benchmark.

## Future Work

Possible extensions include:

- Comparing against transformer-based models such as BERT
- Hyperparameter tuning
- Testing additional feature-selection techniques
- Error analysis of misclassified reviews
- Building a web interface for real-time sentiment prediction
- Evaluating the model on reviews from sources other than IMDb

