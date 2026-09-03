# IMDb Sentiment Analysis with TF-IDF and χ² Feature Selection

## Overview

This project explores sentiment classification of IMDb movie reviews using
traditional machine learning and TF-IDF text representation.

The main focus of the project is not only achieving high classification
accuracy, but also investigating whether the high-dimensional TF-IDF feature
space can be significantly reduced using χ² (Chi-Square) feature selection
without substantially affecting model performance.

The project also includes a FastAPI backend and a simple web interface that
allows users to enter a movie review and receive a real-time sentiment
prediction.

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
- FastAPI inference API
- Web interface for real-time sentiment prediction

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

```text
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
```

The trained model and preprocessing components are saved and used by the
FastAPI backend for inference.

### 1. Text Preprocessing

The reviews are cleaned before feature extraction. The preprocessing includes:

- Converting text to lowercase
- Removing HTML tags
- Removing unnecessary punctuation and special characters
- Normalizing whitespace

The same preprocessing procedure is applied during inference to ensure that
new reviews are processed consistently with the training data.

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

## API & Web Interface

The trained model is integrated into a FastAPI backend that exposes a
prediction endpoint.

The backend performs the following steps:

1. Receives a movie review
2. Applies the same preprocessing used during training
3. Converts the review into TF-IDF features
4. Applies the trained χ² feature selector
5. Passes the selected features to the Logistic Regression model
6. Returns the predicted sentiment

### Prediction Endpoint

```text
POST /predict
```

### Request

```json
{
  "review": "This movie was absolutely fantastic and I loved it."
}
```

### Response

```json
{
  "sentiment": "Positive"
}
```

The project also includes a simple web interface where users can enter a
movie review and receive the predicted sentiment directly in the browser.

## Project Structure

```text
IMDb-Sentiment-Analysis/
│
├── api/
│   └── main.py
│
├── frontend/
│   ├── index.html
│   ├── script.js
│   └── style.css
│
├── imdb_sentiment_analysis.ipynb
│
├── sentiment_model.pkl
├── tfidf_vectorizer.pkl
├── chi2_selector.pkl
│
├── requirements.txt
├── .gitignore
└── README.md
```

### Important Files

| File | Purpose |
|---|---|
| `imdb_sentiment_analysis.ipynb` | Model development, experimentation and evaluation |
| `sentiment_model.pkl` | Trained Logistic Regression model |
| `tfidf_vectorizer.pkl` | Trained TF-IDF vectorizer |
| `chi2_selector.pkl` | Trained χ² feature selector |
| `api/main.py` | FastAPI backend and prediction endpoint |
| `frontend/index.html` | Web interface structure |
| `frontend/script.js` | Frontend API communication |
| `frontend/style.css` | Frontend styling |
| `requirements.txt` | Python dependencies |

## How to Run Locally

### 1. Clone the Repository

```bash
git clone https://github.com/shrey159/IMDb-Sentiment-Analysis.git
cd IMDb-Sentiment-Analysis
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the FastAPI Backend

From the project root directory:

```bash
uvicorn api.main:app --reload
```

The API will run at:

```text
http://127.0.0.1:8000
```

### Swagger API Documentation

FastAPI provides interactive API documentation at:

```text
http://127.0.0.1:8000/docs
```

The `/predict` endpoint can be tested directly through Swagger.

### 4. Start the Frontend

Open another terminal from the project root and run:

```bash
python -m http.server 5500 --bind 127.0.0.1 --directory frontend
```

Then open:

```text
http://127.0.0.1:5500/index.html
```

The frontend communicates with the FastAPI backend and displays the predicted
sentiment for the entered review.

## Key Findings

1. Logistic Regression provided the strongest baseline performance.

2. χ² feature selection reduced the TF-IDF feature space by **73.28%**.

3. Reducing the feature space from 74,849 to 20,000 features resulted in only
   a **0.08 percentage-point decrease in accuracy**.

4. Model performance largely plateaued after approximately 20,000 features.

5. The selected model retains approximately **99.91% of the baseline accuracy**
   while using only **26.72% of the original features**.

6. The results demonstrate that many of the original TF-IDF features provide
   limited additional predictive value.

## Limitations

- The approach uses traditional machine learning rather than transformer-based
  language models.

- The preprocessing pipeline does not explicitly model complex linguistic
  phenomena such as sarcasm or context.

- Mutual Information was explored as an alternative feature-selection method,
  but its computational cost on the high-dimensional TF-IDF representation
  made it impractical for the final benchmark.

- The model was evaluated on IMDb reviews, so its performance may differ on
  reviews from other domains or platforms.

## Future Work

Possible extensions include:

- Comparing against transformer-based models such as BERT

- Hyperparameter tuning

- Testing additional feature-selection techniques

- Error analysis of misclassified reviews

- Evaluating the model on reviews from sources other than IMDb

- Exploring word embeddings and neural-network-based approaches

- Deploying the FastAPI application publicly for real-time sentiment prediction

- Comparing model efficiency in terms of inference time, memory usage, and
  predictive performance

- Investigating whether similar feature-reduction behavior occurs on other
  text classification datasets

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF
- Chi-Square Feature Selection
- Logistic Regression
- Naive Bayes
- Linear SVM
- FastAPI
- Uvicorn
- Pydantic
- Joblib
- HTML
- CSS
- JavaScript
- Jupyter Notebook

## Conclusion

This project demonstrates that effective sentiment classification does not
necessarily require the complete high-dimensional TF-IDF feature space.

Using χ² feature selection, the original 74,849-dimensional TF-IDF
representation was reduced to 20,000 features, resulting in a **73.28%
reduction in dimensionality**.

Despite this substantial reduction, the model achieved an accuracy of
**88.21%**, compared with **88.29%** using the complete feature space.

The difference of only **0.08 percentage points** suggests that a large
portion of the original features contributed limited additional predictive
value for this dataset.

The project therefore provides both a practical sentiment-classification
system and an experimental analysis of the trade-off between feature-space
size and predictive performance.

## Author

**Shreya V**

Computer Science & Engineering

GitHub: [@shrey159](https://github.com/shrey159)