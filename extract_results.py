import joblib
import json
import re
import os
import sys
import numpy as np
import pandas as pd
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("Loading PKL files...")
model = joblib.load(os.path.join(BASE_DIR, "resume_classifier.pkl"))
print(f"  ✅ Model loaded: {type(model).__name__}")

tfidf = joblib.load(os.path.join(BASE_DIR, "tfidf_vectorizer.pkl"))
print(f"  ✅ TF-IDF vectorizer loaded: {tfidf.max_features} max features")

le = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))
print(f"  ✅ Label encoder loaded: {len(le.classes_)} classes")
print(f"     Classes: {list(le.classes_)}")

DATA_PATH = os.path.join(BASE_DIR, "Resume", "Resume.csv")
print(f"\nLoading dataset from {DATA_PATH}...")
df = pd.read_csv(DATA_PATH)

text_col = None
for col in ["Resume_str", "resume_str", "Resume", "resume", "Text", "text"]:
    if col in df.columns:
        text_col = col
        break
if text_col is None:
    for col in df.columns:
        if df[col].dtype == object and col.lower() not in ["category", "id"]:
            text_col = col
            break

print(f"  Text column: {text_col}")
print(f"  Shape: {df.shape}")
print(f"  Categories: {df['Category'].nunique()}")

n_dupes = df.duplicated(subset=[text_col]).sum()
if n_dupes > 0:
    df = df.drop_duplicates(subset=[text_col]).reset_index(drop=True)
    print(f"  Dropped {n_dupes} duplicates → {len(df)} rows")

try:
    import nltk
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.stem import WordNetLemmatizer
    STOP_WORDS = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    HAS_NLTK = True
except Exception:
    HAS_NLTK = False
    STOP_WORDS = set()
    print("  ⚠️  NLTK not available, using basic cleaning")

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if HAS_NLTK:
        tokens = word_tokenize(text)
        tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in STOP_WORDS and len(t) > 2]
    else:
        tokens = [t for t in text.split() if len(t) > 2]
    return " ".join(tokens)

ABBR = {
    "yr": "year", "yrs": "years", "mgmt": "management", "dept": "department",
    "dev": "development", "eng": "engineering", "sr": "senior", "jr": "junior",
    "mgr": "manager", "assoc": "associate", "govt": "government",
    "edu": "education", "info": "information", "tech": "technology",
    "admin": "administration", "exp": "experience",
}

def normalize_text(text: str) -> str:
    tokens = text.split()
    tokens = [ABBR.get(t, t) for t in tokens]
    return " ".join(tokens)

print("\nCleaning text...")
df["clean_text"] = df[text_col].astype(str).apply(clean_text).apply(normalize_text)
print(f"  ✅ Done. Avg clean length: {df['clean_text'].str.len().mean():.0f} chars")

print("\nComputing text statistics...")
df["char_count"] = df[text_col].astype(str).str.len()
df["word_count"] = df[text_col].astype(str).str.split().str.len()
if HAS_NLTK:
    df["sentence_count"] = df[text_col].astype(str).apply(lambda x: len(sent_tokenize(x)))
else:
    df["sentence_count"] = df[text_col].astype(str).str.count(r'[.!?]+')

stats_by_cat = df.groupby("Category").agg(
    mean_chars=("char_count", "mean"),
    mean_words=("word_count", "mean"),
    mean_sentences=("sentence_count", "mean"),
).round(1)

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)

y = le.transform(df["Category"])
X_tfidf = tfidf.transform(df["clean_text"])

X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTrain: {X_train.shape[0]} | Test: {X_test.shape[0]}")

print("\nEvaluating loaded model...")
y_pred = model.predict(X_test)
acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
print(f"  Accuracy:  {acc:.4f}")
print(f"  Precision: {prec:.4f}")
print(f"  Recall:    {rec:.4f}")
print(f"  F1:        {f1:.4f}")

model_name = type(model).__name__
results = {
    model_name: {
        "Accuracy": round(float(acc), 4),
        "Precision": round(float(prec), 4),
        "Recall": round(float(rec), 4),
        "F1": round(float(f1), 4),
        "Time_s": 0
    }
}
all_preds = {model_name: y_pred}

print("\nTraining comparison models...")
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import time

comparison_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced", random_state=42),
    "Multinomial NB": MultinomialNB(alpha=0.1),
    "LinearSVC": LinearSVC(max_iter=5000, C=1.0, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1),
}

loaded_type = type(model).__name__

for name, m in comparison_models.items():
    if name.replace(" ", "") == loaded_type.replace(" ", ""):
        continue
    try:
        t0 = time.time()
        m.fit(X_train, y_train)
        train_time = time.time() - t0
        yp = m.predict(X_test)
        a = accuracy_score(y_test, yp)
        p = precision_score(y_test, yp, average="weighted", zero_division=0)
        r = recall_score(y_test, yp, average="weighted", zero_division=0)
        f = f1_score(y_test, yp, average="weighted", zero_division=0)
        results[name] = {
            "Accuracy": round(float(a), 4),
            "Precision": round(float(p), 4),
            "Recall": round(float(r), 4),
            "F1": round(float(f), 4),
            "Time_s": round(train_time, 2)
        }
        all_preds[name] = yp
        print(f"  {name:25s} | Acc {a:.4f} | F1 {f:.4f} | {train_time:.1f}s")
    except Exception as e:
        print(f"  ⚠️  {name} failed: {e}")

print("\nComputing per-class metrics...")
per_class_f1 = {}
for mname, preds in all_preds.items():
    report = classification_report(y_test, preds, target_names=le.classes_,
                                   output_dict=True, zero_division=0)
    per_class_f1[mname] = {cat: round(report[cat]["f1-score"], 3) for cat in le.classes_}

best_name = max(results, key=lambda k: results[k]["F1"])
best_pred = all_preds[best_name]
cm = confusion_matrix(y_test, best_pred)

confused_pairs = []
cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True) * 100
for i in range(len(le.classes_)):
    for j in range(len(le.classes_)):
        if i != j and cm[i, j] > 0:
            confused_pairs.append({
                "Actual": le.classes_[i],
                "Predicted": le.classes_[j],
                "Count": int(cm[i, j]),
                "Pct_of_actual": round(float(cm_norm[i, j]), 1),
            })
confused_pairs.sort(key=lambda x: x["Count"], reverse=True)

error_rates = []
for i, cat in enumerate(le.classes_):
    cat_mask = y_test == i
    if cat_mask.sum() > 0:
        err = float((best_pred[cat_mask] != y_test[cat_mask]).mean() * 100)
        error_rates.append({
            "Category": cat,
            "Error_Rate": round(err, 1),
            "N_test": int(cat_mask.sum())
        })
error_rates.sort(key=lambda x: x["Error_Rate"], reverse=True)

print("Computing vocabulary stats...")
vocab_stats = []
for cat in sorted(df["Category"].unique()):
    texts = df.loc[df["Category"] == cat, "clean_text"].astype(str)
    all_words = " ".join(texts).lower().split()
    unique_words = set(all_words)
    stop_ratio = sum(1 for w in all_words if w in STOP_WORDS) / max(len(all_words), 1)
    vocab_stats.append({
        "Category": cat,
        "Total_Words": len(all_words),
        "Unique_Words": len(unique_words),
        "TTR": round(len(unique_words) / max(len(all_words), 1), 4),
        "Avg_Word_Length": round(np.mean([len(w) for w in all_words]) if all_words else 0, 2),
        "Stopword_Ratio": round(stop_ratio, 3),
    })
vocab_stats.sort(key=lambda x: x["TTR"], reverse=True)

print("Computing skill matrix...")
SKILLS = {
    "Programming": ["python", "java", "javascript", "sql", "html", "css", "react",
                    "angular", "node", "typescript", "ruby", "php", "swift", "kotlin"],
    "Data/ML": ["machine learning", "deep learning", "data science", "tensorflow",
                "pytorch", "pandas", "numpy", "scikit", "spark", "hadoop", "tableau",
                "power bi", "statistics", "nlp", "big data"],
    "Cloud/DevOps": ["aws", "azure", "gcp", "docker", "kubernetes", "jenkins",
                     "ci cd", "terraform", "ansible", "linux", "devops", "cloud"],
    "Database": ["mysql", "postgresql", "mongodb", "oracle", "redis", "elasticsearch",
                 "database", "nosql"],
    "Soft Skills": ["leadership", "communication", "teamwork", "management",
                    "problem solving", "analytical", "creative", "collaboration",
                    "presentation", "negotiation", "strategic"],
    "Tools/Other": ["excel", "git", "jira", "agile", "scrum", "rest api",
                    "microservices", "figma", "photoshop", "autocad", "sap"],
}

skill_matrix = {}
for domain, skills in SKILLS.items():
    skill_matrix[domain] = {}
    for cat in sorted(df["Category"].unique()):
        cat_text = " ".join(df.loc[df["Category"] == cat, "clean_text"]).lower()
        n = len(df[df["Category"] == cat])
        count = sum(cat_text.count(skill) for skill in skills)
        skill_matrix[domain][cat] = round(count / max(n, 1), 1)

print("Running feature ablation...")
from sklearn.feature_extraction.text import TfidfVectorizer

ablation_results = []
for mf in [500, 1000, 2000, 3000, 5000, 8000, 10000]:
    tv = TfidfVectorizer(max_features=mf, ngram_range=(1, 2), sublinear_tf=True)
    Xt = tv.fit_transform(df["clean_text"])
    Xtr, Xte, ytr, yte = train_test_split(Xt, y, test_size=0.2, random_state=42, stratify=y)
    m = LinearSVC(C=1, class_weight="balanced", max_iter=5000, random_state=42)
    m.fit(Xtr, ytr)
    f1v = f1_score(yte, m.predict(Xte), average="weighted", zero_division=0)
    ablation_results.append({"max_features": mf, "F1": round(float(f1v), 4)})
    print(f"  max_features={mf:>5d} → F1 = {f1v:.4f}")

print("\nBuilding analysis_results.json...")
export_data = {
    "dataset_info": {
        "total_samples": int(len(df)),
        "num_categories": int(df["Category"].nunique()),
        "categories": sorted(list(df["Category"].unique())),
        "category_counts": df["Category"].value_counts().to_dict(),
    },
    "text_statistics": {
        "mean_chars": stats_by_cat["mean_chars"].to_dict(),
        "mean_words": stats_by_cat["mean_words"].to_dict(),
        "mean_sentences": stats_by_cat["mean_sentences"].to_dict(),
    },
    "model_comparison": results,
    "best_model": best_name,
    "per_class_f1": per_class_f1,
    "confusion_data": {
        "top_confused_pairs": confused_pairs[:20],
    },
    "error_rates": error_rates,
    "vocabulary_stats": vocab_stats,
    "skill_matrix": skill_matrix,
    "ablation_results": ablation_results,
}

out_path = os.path.join(BASE_DIR, "analysis_results.json")
with open(out_path, "w") as f:
    json.dump(export_data, f, indent=2, default=str)

print(f"\n✅ Saved: {out_path}")
print(f"   Best model: {best_name} (F1 = {results[best_name]['F1']:.4f})")
print(f"   Models evaluated: {len(results)}")
print(f"   Categories: {len(le.classes_)}")
print("\n🎉 Done! Open the website to see real results.")
