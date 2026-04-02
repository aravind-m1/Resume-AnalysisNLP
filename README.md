# Resume Classification & Analysis Dashboard

An end-to-end Machine Learning pipeline and interactive web dashboard for classifying Resumes into 24 distinct job categories.

## Features

- **Machine Learning Pipeline**: Trained multiple models (Logistic Regression, Multinomial Naive Bayes, Linear SVC, Random Forest, XGBoost) on the Kaggle Resume Dataset.
- **Interactive Web Dashboard**: Built with vanilla HTML/CSS/JS and Chart.js to visualize the model performance metrics dynamically.
- **Detailed Analytics**:
  - Model Performance Comparison (F1, Accuracy, Precision, Recall)
  - Confusion Matrix Analysis & Misclassification Rates
  - Category Distribution
  - Skill Matrix Heatmap
  - Linguistic & Vocabulary Statistics
- **Premium UI**: Modern dark theme with glassmorphism aesthetics.

## Project Structure

- `index.html`, `index.css`, `app.js` — The frontend dashboard
- `analysis_results.json` — The exported model metrics that power the frontend charts
- `resume_analysis_complete.ipynb` / `.py` — The core data science pipeline (EDA, Training, Validation)
- `extract_results.py` — Script to load trained `.pkl` artifacts, evaluate models against testing data, and generate the `analysis_results.json` structure
- `Resume/` — The dataset
- `*.pkl` — Serialized artifacts (LabelEncoder, TF-IDF Vectorizer, Trained Model)

## How to View the Dashboard Locally

1. Clone the repository.
2. Navigate to the project folder.
3. Start a local Python server:
   ```bash
   python -m http.server 8765
   ```
4. Open your browser and navigate to `http://localhost:8765/index.html`

## How to Update Model Results

If you adjust the training pipeline and generate new `*.pkl` models, simply run:
```bash
python extract_results.py
```
This will automatically evaluate the new model against baseline models and update `analysis_results.json` for the dashboard.
