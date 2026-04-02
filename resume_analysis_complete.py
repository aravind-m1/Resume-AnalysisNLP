"""
Generator script for comprehensive resume dataset analysis notebook.
Run: python resume_analysis_complete.py
Produces: resume_analysis_complete.ipynb
"""
import json

def md(source):
    if isinstance(source, str):
        source = source.split("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in source[:-1]] + [source[-1]]}

def code(source):
    if isinstance(source, str):
        source = source.split("\n")
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": [l + "\n" for l in source[:-1]] + [source[-1]]}

cells = []

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: PROJECT OVERVIEW & SETUP
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
# 📊 Comprehensive Resume Dataset Analysis & Classification

## 1. Project Overview & Setup

This notebook provides an **exhaustive analysis** of the Resume Dataset, covering:
- **Deep Exploratory Data Analysis** with multiple groupings and cross-tabulations
- **Text statistics & linguistic analysis** per category
- **N-gram frequency analysis** and skill extraction
- **Category similarity & clustering**
- **Complete model training, testing, and evaluation**
- **Per-class performance breakdowns** and error analysis
- **Results exported** as JSON/CSV for website integration

**Dataset**: Kaggle Resume Dataset (~2400 resumes, 24 job categories)"""))

cells.append(code("""\
# ── 1.0 Install & Import ────────────────────────────────────────────────────
# !pip install -q gensim lime wordcloud scikit-learn pandas matplotlib seaborn nltk

import warnings, os, re, time, random, json, csv
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from itertools import combinations

import nltk
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix,
                             roc_curve, auc, precision_recall_curve, average_precision_score)
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import dendrogram, linkage
from gensim.models import Word2Vec

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import (Embedding, LSTM, Dense, Dropout,
                                         Bidirectional, GlobalMaxPooling1D)
    from tensorflow.keras.preprocessing.text import Tokenizer as KerasTokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.callbacks import EarlyStopping
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("⚠️  TensorFlow not found — LSTM section will be skipped.")

try:
    from lime.lime_text import LimeTextExplainer
    HAS_LIME = True
except ImportError:
    HAS_LIME = False

try:
    from wordcloud import WordCloud
    HAS_WC = True
except ImportError:
    HAS_WC = False

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
if HAS_TF:
    tf.random.set_seed(SEED)

sns.set_style("whitegrid")
plt.rcParams.update({"figure.dpi": 100, "font.size": 11})
print("✅ All imports successful.")"""))

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: BASIC DATASET OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
---
## 2. Basic Dataset Overview

Initial exploration: shape, data types, nulls, duplicates, and sample records."""))

cells.append(code("""\
DATA_PATH = "Resume/Resume.csv"
df = pd.read_csv(DATA_PATH)
text_col = "Resume_str"  # Verified: CSV columns are ID, Resume_str, Resume_html, Category

print(f"{'Shape':<20}: {df.shape}")
print(f"{'Columns':<20}: {list(df.columns)}")
print(f"{'Text column':<20}: {text_col}")
print(f"{'Null values':<20}:")
print(df.isnull().sum())
print(f"\\n{'Duplicate rows':<20}: {df.duplicated().sum()}")
print(f"{'Unique categories':<20}: {df['Category'].nunique()}")
n_dupes = df.duplicated(subset=[text_col]).sum()
if n_dupes > 0:
    print(f"\nDropping {n_dupes} duplicate resumes...")
    df = df.drop_duplicates(subset=[text_col]).reset_index(drop=True)
    print(f"Shape after dedup: {df.shape}")
df.head(3)"""))

cells.append(code("""\
# Data types and memory usage
print("── Data Types & Memory ──")
print(df.dtypes)
print(f"\\nMemory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"\\nBasic statistics for text length:")
df["_raw_len"] = df[text_col].astype(str).str.len()
print(df.groupby("Category")["_raw_len"].describe().round(0).to_string())"""))

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: CATEGORY DISTRIBUTION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
---
## 3. Category Distribution Analysis

Analyzing how resumes are distributed across 24 job categories — frequency counts,
percentages, and tier-based grouping."""))

cells.append(code("""\
# ── 3.1 Frequency table with percentages ─────────────────────────────────────
cat_counts = df["Category"].value_counts()
cat_pct = (cat_counts / len(df) * 100).round(2)
cat_table = pd.DataFrame({"Count": cat_counts, "Percentage": cat_pct,
                           "Cumulative%": cat_pct.cumsum().round(2)})
print("── Category Distribution Table ──")
print(cat_table.to_string())
print(f"\\nTotal resumes: {len(df)}")"""))

cells.append(code("""\
# ── 3.2 Horizontal bar chart ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(18, 8))

# Bar chart
order = cat_counts.index
colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(order)))
bars = axes[0].barh(range(len(order)), cat_counts.values, color=colors)
axes[0].set_yticks(range(len(order)))
axes[0].set_yticklabels(order)
axes[0].set_xlabel("Count")
axes[0].set_title("Resume Count per Category", fontsize=14, fontweight="bold")
for i, (v, p) in enumerate(zip(cat_counts.values, cat_pct.values)):
    axes[0].text(v + 1, i, f"{v} ({p}%)", va="center", fontsize=9)
axes[0].invert_yaxis()

# Donut chart
top_n = 8
top_cats = cat_counts.head(top_n)
other = cat_counts.iloc[top_n:].sum()
donut_data = pd.concat([top_cats, pd.Series({"Others": other})])
wedges, texts, autotexts = axes[1].pie(
    donut_data, labels=donut_data.index, autopct="%1.1f%%",
    startangle=90, pctdistance=0.8,
    colors=plt.cm.Set3(np.linspace(0, 1, len(donut_data))))
centre = plt.Circle((0, 0), 0.55, fc="white")
axes[1].add_artist(centre)
axes[1].set_title("Category Proportions (Top 8 + Others)", fontsize=14, fontweight="bold")

plt.tight_layout()
plt.show()"""))

cells.append(code("""\
# ── 3.3 Tier-based grouping ──────────────────────────────────────────────────
def assign_tier(count):
    if count >= 100:
        return "High (≥100)"
    elif count >= 50:
        return "Medium (50-99)"
    else:
        return "Low (<50)"

tier_df = pd.DataFrame({"Category": cat_counts.index, "Count": cat_counts.values})
tier_df["Tier"] = tier_df["Count"].apply(assign_tier)

print("── Categories Grouped by Sample Size Tier ──\\n")
for tier in ["High (≥100)", "Medium (50-99)", "Low (<50)"]:
    subset = tier_df[tier_df["Tier"] == tier]
    print(f"  {tier}: {len(subset)} categories")
    for _, row in subset.iterrows():
        print(f"    • {row['Category']}: {row['Count']} samples")
    print()

print("── Tier Summary ──")
print(tier_df.groupby("Tier")["Count"].agg(["count", "sum", "mean"]).round(1))"""))

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: TEXT LENGTH ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
---
## 4. Text Length Analysis by Category

Analyzing character count, word count, and sentence count distributions across categories."""))

cells.append(code("""\
# ── 4.1 Compute text statistics ──────────────────────────────────────────────
df["char_count"] = df[text_col].astype(str).str.len()
df["word_count"] = df[text_col].astype(str).str.split().str.len()
df["sentence_count"] = df[text_col].astype(str).apply(lambda x: len(sent_tokenize(x)))
df["avg_word_len"] = df[text_col].astype(str).apply(
    lambda x: np.mean([len(w) for w in x.split()]) if len(x.split()) > 0 else 0)

stats_by_cat = df.groupby("Category").agg(
    mean_chars=("char_count", "mean"),
    median_chars=("char_count", "median"),
    std_chars=("char_count", "std"),
    mean_words=("word_count", "mean"),
    median_words=("word_count", "median"),
    mean_sentences=("sentence_count", "mean"),
    mean_word_len=("avg_word_len", "mean"),
).round(1)

print("── Text Statistics by Category ──")
print(stats_by_cat.sort_values("mean_words", ascending=False).to_string())"""))

cells.append(code("""\
# ── 4.2 Box plots ────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(22, 8))
order = df.groupby("Category")["word_count"].median().sort_values(ascending=False).index

for ax, col, title in zip(axes,
    ["word_count", "char_count", "sentence_count"],
    ["Word Count", "Character Count", "Sentence Count"]):
    sns.boxplot(x=col, y="Category", data=df, order=order, ax=ax,
                palette="coolwarm", fliersize=2)
    ax.set_title(f"{title} by Category", fontsize=13, fontweight="bold")
    ax.set_ylabel("")

plt.tight_layout()
plt.show()"""))

cells.append(code("""\
# ── 4.3 Violin plot for word count ───────────────────────────────────────────
plt.figure(figsize=(18, 8))
top10 = df.groupby("Category")["word_count"].median().nlargest(10).index
df_top10 = df[df["Category"].isin(top10)]
sns.violinplot(x="Category", y="word_count", data=df_top10,
               order=top10, palette="muted", inner="box")
plt.xticks(rotation=45, ha="right")
plt.title("Word Count Distribution — Top 10 Categories (by median)", fontsize=14, fontweight="bold")
plt.ylabel("Word Count")
plt.tight_layout()
plt.show()"""))

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: VOCABULARY & LINGUISTIC ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
---
## 5. Vocabulary & Linguistic Analysis

Measuring vocabulary richness, unique word counts, and linguistic patterns per category."""))

cells.append(code("""\
# ── 5.1 Vocabulary metrics per category ──────────────────────────────────────
STOP_WORDS = set(stopwords.words("english"))

vocab_stats = []
for cat in df["Category"].unique():
    texts = df.loc[df["Category"] == cat, text_col].astype(str)
    all_words = " ".join(texts).lower().split()
    unique_words = set(all_words)
    stop_ratio = sum(1 for w in all_words if w in STOP_WORDS) / max(len(all_words), 1)
    vocab_stats.append({
        "Category": cat,
        "Total_Words": len(all_words),
        "Unique_Words": len(unique_words),
        "TTR": round(len(unique_words) / max(len(all_words), 1), 4),
        "Avg_Word_Length": round(np.mean([len(w) for w in all_words]), 2),
        "Stopword_Ratio": round(stop_ratio, 3),
    })

vocab_df = pd.DataFrame(vocab_stats).sort_values("TTR", ascending=False)
print("── Vocabulary Metrics by Category ──")
print(vocab_df.to_string(index=False))"""))

cells.append(code("""\
# ── 5.2 Vocabulary richness visualization ────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
v = vocab_df.sort_values("TTR", ascending=True)

axes[0].barh(v["Category"], v["TTR"], color=plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(v))))
axes[0].set_title("Type-Token Ratio (Vocabulary Richness)", fontsize=12, fontweight="bold")
axes[0].set_xlabel("TTR")

axes[1].barh(v["Category"], v["Unique_Words"],
             color=plt.cm.Blues(np.linspace(0.3, 0.9, len(v))))
axes[1].set_title("Unique Word Count per Category", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Unique Words")

axes[2].barh(v["Category"], v["Stopword_Ratio"],
             color=plt.cm.Oranges(np.linspace(0.3, 0.9, len(v))))
axes[2].set_title("Stopword Ratio per Category", fontsize=12, fontweight="bold")
axes[2].set_xlabel("Ratio")

plt.tight_layout()
plt.show()"""))

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: N-GRAM FREQUENCY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
---
## 6. N-gram Frequency Analysis

Analyzing the most frequent unigrams, bigrams, and trigrams both overall and per category.
This reveals domain-specific vocabulary and common phrases."""))

cells.append(code("""\
# ── 6.0 Preprocessing for n-gram analysis ────────────────────────────────────
lemmatizer = WordNetLemmatizer()

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\\S+|www\\.\\S+", " ", text)
    text = re.sub(r"\\S+@\\S+", " ", text)
    text = re.sub(r"[^a-z\\s]", " ", text)
    text = re.sub(r"\\s+", " ", text).strip()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(t) for t in tokens
              if t not in STOP_WORDS and len(t) > 2]
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
    tokens = [re.sub(r"(.)\\1{2,}", r"\\1\\1", t) for t in tokens]
    return " ".join(tokens)

print("Cleaning text — this may take 1-2 minutes …")
df["clean_text"] = df[text_col].astype(str).apply(clean_text).apply(normalize_text)
print(f"✅ Done. Avg clean length: {df['clean_text'].str.len().mean():.0f} chars")"""))

cells.append(code("""\
# ── 6.1 Overall top n-grams ──────────────────────────────────────────────────
from sklearn.feature_extraction.text import CountVectorizer

fig, axes = plt.subplots(1, 3, figsize=(22, 8))

for ax, ngram, title in zip(axes,
    [(1,1), (2,2), (3,3)],
    ["Top 20 Unigrams", "Top 20 Bigrams", "Top 15 Trigrams"]):
    n = 15 if ngram[0] == 3 else 20
    cv = CountVectorizer(ngram_range=ngram, max_features=n)
    counts = cv.fit_transform(df["clean_text"])
    freq = dict(zip(cv.get_feature_names_out(), counts.sum(axis=0).A1))
    sorted_f = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    words, vals = zip(*sorted_f)
    ax.barh(range(len(words)), vals, color=plt.cm.plasma(np.linspace(0.2, 0.8, len(words))))
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Frequency")

plt.tight_layout()
plt.show()"""))

cells.append(code("""\
# ── 6.2 Top bigrams per category (selected) ─────────────────────────────────
selected_cats = df["Category"].value_counts().head(8).index.tolist()
fig, axes = plt.subplots(2, 4, figsize=(24, 12))

for ax, cat in zip(axes.flatten(), selected_cats):
    subset = df.loc[df["Category"] == cat, "clean_text"]
    cv = CountVectorizer(ngram_range=(2,2), max_features=10)
    counts = cv.fit_transform(subset)
    freq = dict(zip(cv.get_feature_names_out(), counts.sum(axis=0).A1))
    sorted_f = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    if sorted_f:
        words, vals = zip(*sorted_f)
        ax.barh(range(len(words)), vals, color="steelblue")
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words, fontsize=9)
        ax.invert_yaxis()
    ax.set_title(cat, fontsize=11, fontweight="bold")

plt.suptitle("Top 10 Bigrams per Category", fontsize=16, fontweight="bold", y=1.01)
plt.tight_layout()
plt.show()"""))

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: SKILL/KEYWORD EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
---
## 7. Skill / Keyword Extraction by Category

Using a predefined skill dictionary to detect and quantify skills across categories.
This reveals which domains are programming-heavy vs. soft-skill-heavy."""))

cells.append(code("""\
# ── 7.1 Skill dictionary ────────────────────────────────────────────────────
SKILLS = {
    "Programming": ["python", "java", "javascript", "sql", "html", "css", "react",
                    "angular", "node", "typescript", "ruby", "php", "swift", "kotlin",
                    "scala", "matlab", "perl", "bash", "shell"],
    "Data/ML": ["machine learning", "deep learning", "data science", "tensorflow",
                "pytorch", "pandas", "numpy", "scikit", "spark", "hadoop", "tableau",
                "power bi", "statistics", "nlp", "computer vision", "big data"],
    "Cloud/DevOps": ["aws", "azure", "gcp", "docker", "kubernetes", "jenkins",
                     "ci cd", "terraform", "ansible", "linux", "devops", "cloud"],
    "Database": ["mysql", "postgresql", "mongodb", "oracle", "redis", "elasticsearch",
                 "cassandra", "dynamodb", "database", "nosql"],
    "Soft Skills": ["leadership", "communication", "teamwork", "management",
                    "problem solving", "analytical", "creative", "collaboration",
                    "presentation", "negotiation", "strategic"],
    "Tools/Other": ["excel", "git", "jira", "agile", "scrum", "rest api",
                    "microservices", "figma", "photoshop", "autocad", "sap"],
}

# Count skills per category
skill_matrix = {}
for cat in df["Category"].unique():
    cat_text = " ".join(df.loc[df["Category"] == cat, "clean_text"]).lower()
    skill_matrix[cat] = {}
    for domain, skills in SKILLS.items():
        count = sum(cat_text.count(skill) for skill in skills)
        skill_matrix[cat][domain] = count

skill_df = pd.DataFrame(skill_matrix).T
# Normalize by number of resumes in each category
for cat in skill_df.index:
    n = len(df[df["Category"] == cat])
    skill_df.loc[cat] = (skill_df.loc[cat] / n).round(1)

print("── Average Skill Mentions per Resume by Category ──")
print(skill_df.to_string())"""))

cells.append(code("""\
# ── 7.2 Skill frequency heatmap ─────────────────────────────────────────────
plt.figure(figsize=(14, 12))
sns.heatmap(skill_df, annot=True, fmt=".1f", cmap="YlOrRd", linewidths=0.5)
plt.title("Skill Domain Frequency per Resume (Normalized)", fontsize=14, fontweight="bold")
plt.xlabel("Skill Domain")
plt.ylabel("Category")
plt.tight_layout()
plt.show()"""))

cells.append(code("""\
# ── 7.3 Skill domain profile per category (stacked bar) ─────────────────────
skill_pct = skill_df.div(skill_df.sum(axis=1), axis=0).fillna(0)
skill_pct_sorted = skill_pct.loc[skill_pct.sum(axis=1).sort_values(ascending=True).index]

skill_pct_sorted.plot(kind="barh", stacked=True, figsize=(14, 10),
                      colormap="Set2", edgecolor="white", linewidth=0.5)
plt.title("Skill Domain Profile per Category (Proportional)", fontsize=14, fontweight="bold")
plt.xlabel("Proportion")
plt.legend(title="Skill Domain", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.show()

# Classify categories by dominant skill
dominant = skill_df.idxmax(axis=1)
print("\\n── Dominant Skill Domain per Category ──")
for cat, dom in dominant.items():
    print(f"  {cat:30s} → {dom}")"""))

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: CATEGORY SIMILARITY & OVERLAP
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
---
## 8. Category Similarity & Overlap

Using TF-IDF representations to measure cosine similarity between categories,
perform hierarchical clustering, and visualize in 2D space."""))

cells.append(code("""\
# ── 8.1 Category-level TF-IDF & cosine similarity ───────────────────────────
# Aggregate text per category
cat_texts = df.groupby("Category")["clean_text"].apply(" ".join)
tfidf_cat = TfidfVectorizer(max_features=3000, ngram_range=(1,2), sublinear_tf=True)
cat_vecs = tfidf_cat.fit_transform(cat_texts)

sim_matrix = cosine_similarity(cat_vecs)
sim_df = pd.DataFrame(sim_matrix, index=cat_texts.index, columns=cat_texts.index)

plt.figure(figsize=(16, 14))
mask = np.triu(np.ones_like(sim_df, dtype=bool), k=1)
sns.heatmap(sim_df, annot=True, fmt=".2f", cmap="RdYlBu_r",
            mask=mask, vmin=0, vmax=1, linewidths=0.3)
plt.title("Category Cosine Similarity (TF-IDF)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.show()

# Top similar pairs
pairs = []
for i in range(len(sim_df)):
    for j in range(i+1, len(sim_df)):
        pairs.append((sim_df.index[i], sim_df.columns[j], sim_df.iloc[i, j]))
pairs.sort(key=lambda x: x[2], reverse=True)
print("\\n── Top 10 Most Similar Category Pairs ──")
for c1, c2, s in pairs[:10]:
    print(f"  {c1:30s} ↔ {c2:30s} : {s:.4f}")"""))

cells.append(code("""\
# ── 8.2 Hierarchical clustering dendrogram ───────────────────────────────────
Z = linkage(cat_vecs.toarray(), method="ward")

plt.figure(figsize=(16, 8))
dendrogram(Z, labels=cat_texts.index, leaf_rotation=45, leaf_font_size=10,
           color_threshold=3.0)
plt.title("Hierarchical Clustering of Resume Categories", fontsize=14, fontweight="bold")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()"""))

cells.append(code("""\
# ── 8.3 t-SNE 2D projection (document level) ────────────────────────────────
tfidf_all = TfidfVectorizer(max_features=2000, sublinear_tf=True)
X_all = tfidf_all.fit_transform(df["clean_text"])

# PCA first for speed, then t-SNE
pca_50 = PCA(n_components=50, random_state=SEED)
X_pca = pca_50.fit_transform(X_all.toarray())
tsne = TSNE(n_components=2, random_state=SEED, perplexity=30, n_iter=1000)
X_2d = tsne.fit_transform(X_pca)

plt.figure(figsize=(16, 12))
cats = df["Category"].values
unique_cats = sorted(df["Category"].unique())
cmap = plt.cm.get_cmap("tab20", len(unique_cats))
for i, cat in enumerate(unique_cats):
    mask = cats == cat
    plt.scatter(X_2d[mask, 0], X_2d[mask, 1], c=[cmap(i)], label=cat,
                alpha=0.6, s=20, edgecolors="none")
plt.title("t-SNE Projection of Resumes (TF-IDF)", fontsize=14, fontweight="bold")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8, ncol=1)
plt.tight_layout()
plt.show()"""))

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9: CROSS-TABULATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
---
## 9. Cross-Tabulation Analysis

Grouping categories by multiple dimensions: resume length bins, skill-domain profiles,
and vocabulary richness to reveal structural patterns."""))

cells.append(code("""\
# ── 9.1 Category × Resume Length Bins ────────────────────────────────────────
df["length_bin"] = pd.cut(df["word_count"],
    bins=[0, 200, 500, 1000, float("inf")],
    labels=["Short (<200)", "Medium (200-500)", "Long (500-1K)", "Very Long (>1K)"])

cross_len = pd.crosstab(df["Category"], df["length_bin"], normalize="index").round(3) * 100

plt.figure(figsize=(14, 10))
sns.heatmap(cross_len, annot=True, fmt=".1f", cmap="YlGnBu", linewidths=0.3)
plt.title("Category × Resume Length Distribution (%)", fontsize=14, fontweight="bold")
plt.xlabel("Resume Length Bin")
plt.ylabel("Category")
plt.tight_layout()
plt.show()

print("\\n📌 Key observations:")
short_heavy = cross_len["Short (<200)"].nlargest(3)
long_heavy = cross_len["Very Long (>1K)"].nlargest(3)
print(f"  • Shortest resumes: {', '.join(short_heavy.index)}")
print(f"  • Longest resumes: {', '.join(long_heavy.index)}")"""))

cells.append(code("""\
# ── 9.2 Category × Dominant Skill Domain ─────────────────────────────────────
# Using per-resume skill detection
def get_dominant_skill(text):
    text_lower = text.lower()
    scores = {}
    for domain, skills in SKILLS.items():
        scores[domain] = sum(text_lower.count(s) for s in skills)
    return max(scores, key=scores.get) if max(scores.values()) > 0 else "None"

df["dominant_skill"] = df["clean_text"].apply(get_dominant_skill)
cross_skill = pd.crosstab(df["Category"], df["dominant_skill"], normalize="index").round(3) * 100

plt.figure(figsize=(14, 10))
sns.heatmap(cross_skill, annot=True, fmt=".1f", cmap="PuBuGn", linewidths=0.3)
plt.title("Category × Dominant Skill Domain (%)", fontsize=14, fontweight="bold")
plt.xlabel("Dominant Skill Domain")
plt.ylabel("Category")
plt.tight_layout()
plt.show()"""))

cells.append(code("""\
# ── 9.3 Multi-dimensional summary ───────────────────────────────────────────
summary = df.groupby("Category").agg(
    n_resumes=("Category", "count"),
    avg_words=("word_count", "mean"),
    avg_sentences=("sentence_count", "mean"),
    pct_programming=("dominant_skill", lambda x: (x == "Programming").mean() * 100),
    pct_data_ml=("dominant_skill", lambda x: (x == "Data/ML").mean() * 100),
    pct_soft_skills=("dominant_skill", lambda x: (x == "Soft Skills").mean() * 100),
).round(1)
summary = summary.sort_values("n_resumes", ascending=False)
print("── Multi-Dimensional Category Summary ──")
print(summary.to_string())"""))

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10: WORD CLOUD GALLERY
# ═══════════════════════════════════════════════════════════════════════════════
cells.append(md("""\
---
## 10. Word Cloud Gallery

Individual word clouds for every category, revealing the most prominent vocabulary."""))

cells.append(code("""\
# ── 10.1 Word clouds for all categories ──────────────────────────────────────
if HAS_WC:
    all_cats = sorted(df["Category"].unique())
    n_cats = len(all_cats)
    ncols = 5
    nrows = (n_cats + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(25, nrows * 4))
    axes_flat = axes.flatten()

    for i, cat in enumerate(all_cats):
        text = " ".join(df.loc[df["Category"] == cat, "clean_text"])
        wc = WordCloud(width=400, height=250, background_color="white",
                       colormap="viridis", max_words=60).generate(text)
        axes_flat[i].imshow(wc, interpolation="bilinear")
        axes_flat[i].set_title(cat, fontsize=10, fontweight="bold")
        axes_flat[i].axis("off")

    # Hide unused axes
    for j in range(i + 1, len(axes_flat)):
        axes_flat[j].axis("off")

    plt.suptitle("Word Clouds — All Categories", fontsize=18, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.show()
else:
    print("wordcloud not installed — skipping.")"""))

# ═══════════════════════════════════════════════════════════════════════════════
# PART B — MODEL TRAINING & TESTING (Sections 11-14)
# ═══════════════════════════════════════════════════════════════════════════════

# SECTION 11: DATA PREPROCESSING & FEATURE ENGINEERING
cells.append(md("""\
---
# Part B — Model Training & Testing

## 11. Data Preprocessing & Feature Engineering

Preparing features for model training: TF-IDF vectorization, Word2Vec embeddings,
label encoding, and stratified train-test split."""))

cells.append(code("""\
# ── 11.1 TF-IDF Vectorization ───────────────────────────────────────────────
tfidf = TfidfVectorizer(max_features=8000, ngram_range=(1, 2), sublinear_tf=True)
X_tfidf = tfidf.fit_transform(df["clean_text"])
print(f"TF-IDF matrix shape: {X_tfidf.shape}")"""))

cells.append(code("""\
# ── 11.2 Word2Vec Document Vectors ──────────────────────────────────────────
tokenized = df["clean_text"].apply(str.split).tolist()
w2v_model = Word2Vec(sentences=tokenized, vector_size=300, window=7,
                     min_count=2, workers=4, seed=SEED, epochs=20)

def doc_vector(tokens, model, size=300):
    vecs = [model.wv[t] for t in tokens if t in model.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(size)

X_w2v = np.vstack(df["clean_text"].apply(str.split).apply(
    lambda t: doc_vector(t, w2v_model)).values)
print(f"Word2Vec matrix shape: {X_w2v.shape}")"""))

cells.append(code("""\
# ── 11.3 Label Encoding & Stratified Split ──────────────────────────────────
le = LabelEncoder()
y = le.fit_transform(df["Category"])
print(f"Classes ({len(le.classes_)}): {list(le.classes_)}")

X_train_tf, X_test_tf, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=SEED, stratify=y)
X_train_w2v, X_test_w2v, _, _ = train_test_split(
    X_w2v, y, test_size=0.2, random_state=SEED, stratify=y)

print(f"Train: {X_train_tf.shape[0]} | Test: {X_test_tf.shape[0]}")
print(f"\\nTraining set class distribution:")
train_dist = pd.Series(y_train).map(dict(enumerate(le.classes_))).value_counts()
print(train_dist.to_string())"""))

# SECTION 12: MODEL TRAINING
cells.append(md("""\
---
## 12. Model Training

Training 5 classical ML models + BiLSTM deep learning model."""))

cells.append(code("""\
# ── 12.1 Train classical models ─────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000, C=1.0, class_weight="balanced", random_state=SEED),
    "Multinomial NB": MultinomialNB(alpha=0.1),
    "LinearSVC": LinearSVC(
        max_iter=5000, C=1.0, class_weight="balanced", random_state=SEED),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=None, class_weight="balanced",
        random_state=SEED, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=5, random_state=SEED),
}

# Add XGBoost if available
if HAS_XGB:
    models["XGBoost"] = XGBClassifier(
        n_estimators=300, learning_rate=0.1, max_depth=6,
        eval_metric="mlogloss",
        random_state=SEED, n_jobs=-1)
else:
    print("XGBoost not installed — skipping.")

results = {}
all_preds = {}

for name, model in models.items():
    t0 = time.time()
    model.fit(X_train_tf, y_train)
    train_time = time.time() - t0
    y_pred = model.predict(X_test_tf)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    results[name] = {"Accuracy": acc, "Precision": prec, "Recall": rec,
                     "F1": f1, "Time_s": round(train_time, 2)}
    all_preds[name] = y_pred
    print(f"{name:25s} | Acc {acc:.4f} | F1 {f1:.4f} | {train_time:.1f}s")

# ── 12.1b Cross-validation scores ───────────────────────────────────────
print("\\n── Stratified 5-Fold Cross-Validation ──")
from sklearn.model_selection import cross_val_score
for name, model_obj in models.items():
    cv_scores = cross_val_score(model_obj, X_tfidf, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                                scoring="f1_weighted", n_jobs=-1)
    print(f"  {name:25s} | CV F1 = {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print("\\n✅ All classical models trained + cross-validated.")"""))

cells.append(code("""\
# ── 12.2 Train BiLSTM ───────────────────────────────────────────────────────
if HAS_TF:
    MAX_WORDS, MAX_LEN = 10000, 300
    keras_tok = KerasTokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    keras_tok.fit_on_texts(df["clean_text"])

    texts_train, texts_test, yl_train, yl_test = train_test_split(
        df["clean_text"].values, y, test_size=0.2, random_state=SEED, stratify=y)
    X_seq_train = pad_sequences(keras_tok.texts_to_sequences(texts_train), maxlen=MAX_LEN)
    X_seq_test  = pad_sequences(keras_tok.texts_to_sequences(texts_test),  maxlen=MAX_LEN)

    lstm_model = Sequential([
        Embedding(MAX_WORDS, 128, input_length=MAX_LEN),
        Bidirectional(LSTM(128, return_sequences=True)),
        GlobalMaxPooling1D(),
        Dense(128, activation="relu"),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(len(le.classes_), activation="softmax"),
    ])
    lstm_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    es = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
    t0 = time.time()
    history = lstm_model.fit(X_seq_train, yl_train, validation_split=0.1,
                             epochs=15, batch_size=32, callbacks=[es], verbose=1)
    train_time = time.time() - t0

    y_pred_lstm = np.argmax(lstm_model.predict(X_seq_test), axis=1)
    acc  = accuracy_score(yl_test, y_pred_lstm)
    prec = precision_score(yl_test, y_pred_lstm, average="weighted", zero_division=0)
    rec  = recall_score(yl_test, y_pred_lstm, average="weighted", zero_division=0)
    f1   = f1_score(yl_test, y_pred_lstm, average="weighted", zero_division=0)
    results["BiLSTM"] = {"Accuracy": acc, "Precision": prec, "Recall": rec,
                         "F1": f1, "Time_s": round(train_time, 2)}
    all_preds["BiLSTM"] = y_pred_lstm
    print(f"\\nBiLSTM | Acc {acc:.4f} | F1 {f1:.4f} | {train_time:.1f}s")
else:
    print("Skipping LSTM — TensorFlow not available.")"""))

# SECTION 13: HYPERPARAMETER TUNING
cells.append(md("""\
---
## 13. Hyperparameter Tuning

Using RandomizedSearchCV with stratified K-Fold to fine-tune top models."""))

cells.append(code("""\
# ── 13.1 Tune LinearSVC ─────────────────────────────────────────────────────
svc_params = {"C": [0.01, 0.1, 0.5, 1, 2, 5, 10]}
svc_search = RandomizedSearchCV(
    LinearSVC(class_weight="balanced", max_iter=5000, random_state=SEED),
    svc_params, n_iter=5, cv=StratifiedKFold(3, shuffle=True, random_state=SEED),
    scoring="f1_weighted", random_state=SEED, n_jobs=-1)
svc_search.fit(X_train_tf, y_train)
print(f"Best SVC params : {svc_search.best_params_}")
print(f"Best SVC F1 (CV): {svc_search.best_score_:.4f}")

# ── 13.2 Tune Random Forest ────────────────────────────────────────────────
rf_params = {"n_estimators": [200, 300, 500], "max_depth": [None, 30, 50],
             "min_samples_split": [2, 5]}
rf_search = RandomizedSearchCV(
    RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1),
    rf_params, n_iter=6, cv=StratifiedKFold(3, shuffle=True, random_state=SEED),
    scoring="f1_weighted", random_state=SEED, n_jobs=-1)
rf_search.fit(X_train_tf, y_train)
print(f"\\nBest RF params : {rf_search.best_params_}")
print(f"Best RF F1 (CV): {rf_search.best_score_:.4f}")

# ── 13.3 Tune Logistic Regression ───────────────────────────────────────
lr_params = {"C": [0.01, 0.1, 0.5, 1, 2, 5, 10]}
lr_search = RandomizedSearchCV(
    LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
    lr_params, n_iter=5, cv=StratifiedKFold(3, shuffle=True, random_state=SEED),
    scoring="f1_weighted", random_state=SEED, n_jobs=-1)
lr_search.fit(X_train_tf, y_train)
print(f"\nBest LR params : {lr_search.best_params_}")
print(f"Best LR F1 (CV): {lr_search.best_score_:.4f}")

# Evaluate tuned models
for tag, searcher in [("Tuned SVC", svc_search), ("Tuned RF", rf_search), ("Tuned LR", lr_search)]:
    yp = searcher.best_estimator_.predict(X_test_tf)
    acc = accuracy_score(y_test, yp)
    f1  = f1_score(y_test, yp, average="weighted", zero_division=0)
    prec = precision_score(y_test, yp, average="weighted", zero_division=0)
    rec  = recall_score(y_test, yp, average="weighted", zero_division=0)
    results[tag] = {"Accuracy": acc, "Precision": prec, "Recall": rec,
                    "F1": f1, "Time_s": 0}
    all_preds[tag] = yp
    print(f"{tag:15s} | Acc {acc:.4f} | F1 {f1:.4f}")"""))

# SECTION 14: MODEL OPTIMIZATION
cells.append(md("""\
---
## 14. Model Optimization

Feature ablation study and class weighting analysis."""))

cells.append(code("""\
# ── 14.1 Feature ablation: effect of max_features ───────────────────────────
ablation_results = []
for mf in [500, 1000, 2000, 3000, 5000, 8000, 10000]:
    tv = TfidfVectorizer(max_features=mf, ngram_range=(1,2), sublinear_tf=True)
    Xt = tv.fit_transform(df["clean_text"])
    Xtr, Xte, ytr, yte = train_test_split(Xt, y, test_size=0.2, random_state=SEED, stratify=y)
    m = LinearSVC(C=1, class_weight="balanced", max_iter=5000, random_state=SEED)
    m.fit(Xtr, ytr)
    f1v = f1_score(yte, m.predict(Xte), average="weighted", zero_division=0)
    ablation_results.append({"max_features": mf, "F1": round(f1v, 4)})
    print(f"max_features={mf:>5d}  →  F1 = {f1v:.4f}")

# Plot ablation
plt.figure(figsize=(10, 5))
abl_df = pd.DataFrame(ablation_results)
plt.plot(abl_df["max_features"], abl_df["F1"], "o-", linewidth=2, markersize=8, color="teal")
plt.xlabel("max_features")
plt.ylabel("Weighted F1 Score")
plt.title("Feature Ablation Study (LinearSVC)", fontsize=14, fontweight="bold")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()"""))

# ═══════════════════════════════════════════════════════════════════════════════
# PART C — COMPREHENSIVE RESULTS & EVALUATION (Sections 15-22)
# ═══════════════════════════════════════════════════════════════════════════════

# SECTION 15: OVERALL MODEL COMPARISON
cells.append(md("""\
---
# Part C — Comprehensive Results & Evaluation

## 15. Overall Model Comparison

Comparing all trained models across multiple metrics with tables and charts."""))

cells.append(code("""\
# ── 15.1 Results summary table ──────────────────────────────────────────────
res_df = pd.DataFrame(results).T.drop(columns=["Time_s"], errors="ignore")
time_df = pd.DataFrame({k: {"Time_s": v["Time_s"]} for k, v in results.items()}).T
full_res = pd.concat([res_df, time_df], axis=1).sort_values("F1", ascending=False)

print("═" * 75)
print("  COMPLETE MODEL COMPARISON")
print("═" * 75)
print(full_res.to_string())
print("═" * 75)
print(f"\\n🏆 Best model: {full_res.index[0]} (F1 = {full_res['F1'].iloc[0]:.4f})")"""))

cells.append(code("""\
# ── 15.2 Grouped bar chart ─────────────────────────────────────────────────
metrics = ["Accuracy", "Precision", "Recall", "F1"]
plot_data = res_df[metrics].sort_values("F1", ascending=True)

fig, ax = plt.subplots(figsize=(12, 7))
x = np.arange(len(plot_data))
width = 0.18
colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63"]

for i, (metric, color) in enumerate(zip(metrics, colors)):
    bars = ax.barh(x + i * width, plot_data[metric], width, label=metric, color=color, alpha=0.85)

ax.set_yticks(x + width * 1.5)
ax.set_yticklabels(plot_data.index)
ax.set_xlabel("Score")
ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
ax.legend(loc="lower right")
ax.set_xlim(0, 1.1)
plt.tight_layout()
plt.show()"""))

cells.append(code("""\
# ── 15.3 Radar chart for top models ─────────────────────────────────────────
top_models = res_df.nlargest(4, "F1")
categories_radar = metrics
N = len(categories_radar)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
colors_radar = ["#E91E63", "#2196F3", "#4CAF50", "#FF9800"]

for idx, (model_name, row) in enumerate(top_models.iterrows()):
    values = [row[m] for m in categories_radar]
    values += values[:1]
    ax.plot(angles, values, "o-", linewidth=2, label=model_name, color=colors_radar[idx])
    ax.fill(angles, values, alpha=0.1, color=colors_radar[idx])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories_radar, fontsize=11)
ax.set_ylim(0, 1.05)
ax.set_title("Top 4 Models — Radar Comparison", fontsize=14, fontweight="bold", pad=20)
ax.legend(loc="lower right", bbox_to_anchor=(1.3, 0))
plt.tight_layout()
plt.show()"""))

# SECTION 16: PER-CLASS PERFORMANCE
cells.append(md("""\
---
## 16. Per-Class Performance Analysis

Breaking down precision, recall, and F1 for each category across all models."""))

cells.append(code("""\
# ── 16.1 Per-class F1 heatmap ───────────────────────────────────────────────
per_class_f1 = {}
eval_preds = {k: v for k, v in all_preds.items()}

for model_name, preds in eval_preds.items():
    yt = yl_test if model_name == "BiLSTM" and HAS_TF else y_test
    report = classification_report(yt, preds, target_names=le.classes_,
                                   output_dict=True, zero_division=0)
    per_class_f1[model_name] = {cat: report[cat]["f1-score"] for cat in le.classes_}

pcf1_df = pd.DataFrame(per_class_f1).round(3)

plt.figure(figsize=(16, 12))
sns.heatmap(pcf1_df, annot=True, fmt=".2f", cmap="RdYlGn", vmin=0, vmax=1,
            linewidths=0.3)
plt.title("Per-Category F1 Score by Model", fontsize=14, fontweight="bold")
plt.xlabel("Model")
plt.ylabel("Category")
plt.tight_layout()
plt.show()"""))

cells.append(code("""\
# ── 16.2 Best model per category ────────────────────────────────────────────
best_per_cat = pcf1_df.idxmax(axis=1)
best_f1_per_cat = pcf1_df.max(axis=1)

print("── Best Model per Category ──\\n")
for cat in le.classes_:
    bm = best_per_cat[cat]
    bf = best_f1_per_cat[cat]
    print(f"  {cat:30s} → {bm:25s} (F1 = {bf:.3f})")

# Worst categories
print("\\n── Hardest Categories (lowest best-F1) ──\\n")
worst = best_f1_per_cat.nsmallest(5)
for cat, f1v in worst.items():
    print(f"  {cat:30s} → F1 = {f1v:.3f} (best: {best_per_cat[cat]})")"""))

# SECTION 17: CONFUSION MATRIX ANALYSIS
cells.append(md("""\
---
## 17. Confusion Matrix Analysis

Detailed confusion matrix for the best model, normalized version,
and analysis of most-confused category pairs."""))

cells.append(code("""\
# ── 17.1 Full confusion matrix (best model) ─────────────────────────────────
best_name = full_res.index[0]
best_pred = all_preds[best_name]
eval_y_best = yl_test if best_name == "BiLSTM" and HAS_TF else y_test

cm = confusion_matrix(eval_y_best, best_pred)

fig, axes = plt.subplots(1, 2, figsize=(28, 12))

# Raw counts
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_, ax=axes[0])
axes[0].set_title(f"Confusion Matrix — {best_name} (Counts)", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")
axes[0].tick_params(axis="x", rotation=45)

# Normalized (percentage)
cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True) * 100
sns.heatmap(cm_norm, annot=True, fmt=".1f", cmap="YlOrRd",
            xticklabels=le.classes_, yticklabels=le.classes_, ax=axes[1])
axes[1].set_title(f"Confusion Matrix — {best_name} (% Normalized)", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Predicted")
axes[1].set_ylabel("Actual")
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()"""))

cells.append(code("""\
# ── 17.2 Top confused pairs ─────────────────────────────────────────────────
confused_pairs = []
for i in range(len(le.classes_)):
    for j in range(len(le.classes_)):
        if i != j and cm[i, j] > 0:
            confused_pairs.append({
                "Actual": le.classes_[i],
                "Predicted": le.classes_[j],
                "Count": cm[i, j],
                "Pct_of_actual": round(cm_norm[i, j], 1),
            })

confused_df = pd.DataFrame(confused_pairs).sort_values("Count", ascending=False)
print("── Top 15 Most Confused Category Pairs ──\\n")
print(confused_df.head(15).to_string(index=False))

# Bar chart
top_confused = confused_df.head(10)
labels = [f"{r['Actual']}\\n→ {r['Predicted']}" for _, r in top_confused.iterrows()]
plt.figure(figsize=(14, 6))
plt.barh(range(len(labels)), top_confused["Count"].values, color="salmon")
plt.yticks(range(len(labels)), labels, fontsize=9)
plt.xlabel("Misclassification Count")
plt.title("Top 10 Most Confused Category Pairs", fontsize=14, fontweight="bold")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()"""))

# SECTION 18: ERROR ANALYSIS
cells.append(md("""\
---
## 18. Error Analysis

Inspecting misclassified samples to understand failure patterns."""))

cells.append(code("""\
# ── 18.1 Misclassified samples ──────────────────────────────────────────────
misclassified_mask = eval_y_best != best_pred
misclassified_idx = np.where(misclassified_mask)[0]

print(f"Total misclassified: {len(misclassified_idx)} / {len(eval_y_best)} "
      f"({len(misclassified_idx)/len(eval_y_best)*100:.1f}%)\\n")

# Show sample misclassifications
print("── Sample Misclassifications ──\\n")
for i in misclassified_idx[:10]:
    actual = le.classes_[eval_y_best[i]]
    predicted = le.classes_[best_pred[i]]
    print(f"  Sample {i}: {actual} → predicted as {predicted}")

# Error rate per category
print("\\n── Error Rate per Category ──\\n")
error_rates = []
for i, cat in enumerate(le.classes_):
    cat_mask = eval_y_best == i
    if cat_mask.sum() > 0:
        err = (best_pred[cat_mask] != eval_y_best[cat_mask]).mean() * 100
        error_rates.append({"Category": cat, "Error_Rate": round(err, 1),
                           "N_test": int(cat_mask.sum())})

err_df = pd.DataFrame(error_rates).sort_values("Error_Rate", ascending=False)
print(err_df.to_string(index=False))

# Plot
plt.figure(figsize=(12, 8))
colors = ["#E91E63" if e > 20 else "#FF9800" if e > 10 else "#4CAF50"
          for e in err_df["Error_Rate"]]
plt.barh(err_df["Category"], err_df["Error_Rate"], color=colors)
plt.xlabel("Error Rate (%)")
plt.title("Error Rate per Category (Best Model)", fontsize=14, fontweight="bold")
plt.gca().invert_yaxis()
plt.axvline(x=10, color="gray", linestyle="--", alpha=0.5, label="10% threshold")
plt.legend()
plt.tight_layout()
plt.show()"""))

# SECTION 19: ROC & PRECISION-RECALL CURVES
cells.append(md("""\
---
## 19. ROC & Precision-Recall Curves

One-vs-Rest ROC and Precision-Recall curves for the best model."""))

cells.append(code("""\
# ── 19.1 ROC curves (using Logistic Regression for probabilities) ───────────
lr_model = LogisticRegression(max_iter=1000, C=1, class_weight="balanced", random_state=SEED)
lr_model.fit(X_train_tf, y_train)
y_prob = lr_model.predict_proba(X_test_tf)
y_test_bin = label_binarize(y_test, classes=range(len(le.classes_)))

# Plot ROC for top-6 categories by sample count
fig, axes = plt.subplots(1, 2, figsize=(20, 8))
top6_idx = [list(le.classes_).index(c) for c in
            df["Category"].value_counts().head(6).index if c in le.classes_]

colors_roc = plt.cm.Set1(np.linspace(0, 1, len(top6_idx)))
for idx, (cls_idx, color) in enumerate(zip(top6_idx, colors_roc)):
    fpr, tpr, _ = roc_curve(y_test_bin[:, cls_idx], y_prob[:, cls_idx])
    roc_auc = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, color=color, linewidth=2,
                label=f"{le.classes_[cls_idx]} (AUC={roc_auc:.3f})")

axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curves (Top 6 Categories)", fontsize=13, fontweight="bold")
axes[0].legend(fontsize=8, loc="lower right")

# Precision-Recall curves
for idx, (cls_idx, color) in enumerate(zip(top6_idx, colors_roc)):
    prec_vals, rec_vals, _ = precision_recall_curve(y_test_bin[:, cls_idx], y_prob[:, cls_idx])
    ap = average_precision_score(y_test_bin[:, cls_idx], y_prob[:, cls_idx])
    axes[1].plot(rec_vals, prec_vals, color=color, linewidth=2,
                label=f"{le.classes_[cls_idx]} (AP={ap:.3f})")

axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curves (Top 6 Categories)", fontsize=13, fontweight="bold")
axes[1].legend(fontsize=8, loc="lower left")

plt.tight_layout()
plt.show()"""))

# SECTION 20: MODEL INTERPRETABILITY
cells.append(md("""\
---
## 20. Model Interpretability

Top discriminative features per category and LIME explanations."""))

cells.append(code("""\
# ── 20.1 Top TF-IDF features per category (heatmap) ────────────────────────
interp_model = svc_search.best_estimator_
feature_names = np.array(tfidf.get_feature_names_out())

top_n = 5
top_features_data = {}
for i, cat in enumerate(le.classes_):
    if hasattr(interp_model, "coef_"):
        top_idx = interp_model.coef_[i].argsort()[-top_n:][::-1]
        top_features_data[cat] = {feature_names[j]: interp_model.coef_[i][j]
                                  for j in top_idx}

print("── Top 5 Discriminative Words per Category ──\\n")
for cat, features in top_features_data.items():
    words = ", ".join(features.keys())
    print(f"  {cat:30s}: {words}")"""))

cells.append(code("""\
# ── 20.2 LIME explanation ───────────────────────────────────────────────────
if HAS_LIME:
    lr_for_lime = LogisticRegression(max_iter=1000, C=1, class_weight="balanced",
                                     random_state=SEED)
    lr_for_lime.fit(X_train_tf, y_train)

    def lime_predict_proba(texts):
        return lr_for_lime.predict_proba(tfidf.transform(texts))

    explainer = LimeTextExplainer(class_names=le.classes_)
    raw_texts = df[text_col].values
    _, raw_test, _, _ = train_test_split(raw_texts, y, test_size=0.2,
                                         random_state=SEED, stratify=y)

    print("LIME explanation for test sample 0:")
    exp = explainer.explain_instance(raw_test[0], lime_predict_proba,
                                     num_features=10, top_labels=3)
    exp.show_in_notebook(text=False)
else:
    print("LIME not installed — skipping.")"""))

# SECTION 21: PREDICTION PIPELINE
cells.append(md("""\
---
## 21. Prediction Pipeline & Demo

Reusable prediction function with confidence scores."""))

cells.append(code("""\
# ── 21.1 Prediction function with confidence ────────────────────────────────
BEST_MODEL = svc_search.best_estimator_

# For confidence scores, use LogReg
lr_conf = LogisticRegression(max_iter=1000, C=1, class_weight="balanced", random_state=SEED)
lr_conf.fit(X_train_tf, y_train)

def predict_resume(resume_text: str, top_k: int = 3):
    cleaned = normalize_text(clean_text(resume_text))
    vec = tfidf.transform([cleaned])
    pred = BEST_MODEL.predict(vec)[0]
    pred_label = le.inverse_transform([pred])[0]

    # Confidence from LogReg
    probs = lr_conf.predict_proba(vec)[0]
    top_indices = probs.argsort()[-top_k:][::-1]
    top_predictions = [(le.classes_[i], round(probs[i] * 100, 1)) for i in top_indices]

    return pred_label, top_predictions

# Demo
demo_resumes = [
    "Experienced Java developer with 5 years in Spring Boot, microservices, REST APIs, and AWS.",
    "Data scientist skilled in Python, machine learning, deep learning, TensorFlow, and statistics.",
    "HR professional with expertise in talent acquisition, employee engagement, and compliance.",
    "Web designer proficient in Figma, Adobe XD, HTML, CSS, JavaScript, responsive design.",
    "Network engineer experienced in Cisco routers, firewalls, VPN, and network security.",
]

print("── Demo Predictions with Confidence ──\\n")
for r in demo_resumes:
    pred, top3 = predict_resume(r)
    conf_str = " | ".join([f"{cat}: {pct}%" for cat, pct in top3])
    print(f"  ➤ {pred:25s} ← \\"{r[:60]}…\\"")
    print(f"    Top 3: {conf_str}\\n")"""))

# SECTION 22: RESULTS EXPORT
cells.append(md("""\
---
## 22. Results Export for Website

Saving all analysis results as JSON/CSV files for website integration."""))

cells.append(code("""\
# ── 22.1 Export model comparison ────────────────────────────────────────────
import json as json_lib

# Model comparison CSV
full_res.to_csv("model_comparison.csv")
print("✅ Saved: model_comparison.csv")

# Per-class metrics CSV
pcf1_df.to_csv("per_class_metrics.csv")
print("✅ Saved: per_class_metrics.csv")

# Comprehensive results JSON
export_data = {
    "dataset_info": {
        "total_samples": int(len(df)),
        "num_categories": int(df["Category"].nunique()),
        "categories": list(df["Category"].unique()),
        "category_counts": df["Category"].value_counts().to_dict(),
    },
    "text_statistics": stats_by_cat.to_dict(),
    "model_comparison": {
        name: {k: round(float(v), 4) for k, v in metrics_dict.items()}
        for name, metrics_dict in results.items()
    },
    "best_model": full_res.index[0],
    "per_class_f1": pcf1_df.to_dict(),
    "confusion_data": {
        "top_confused_pairs": confused_df.head(20).to_dict(orient="records"),
    },
    "error_rates": err_df.to_dict(orient="records"),
    "vocabulary_stats": vocab_df.to_dict(orient="records"),
    "skill_matrix": skill_df.to_dict(),
    "ablation_results": ablation_results,
}

with open("analysis_results.json", "w") as f:
    json_lib.dump(export_data, f, indent=2, default=str)
print("✅ Saved: analysis_results.json")

# Save model artifacts
import joblib
joblib.dump(BEST_MODEL, "best_resume_model.joblib")
joblib.dump(tfidf, "tfidf_vectorizer.joblib")
joblib.dump(le, "label_encoder.joblib")
print("✅ Saved: best_resume_model.joblib, tfidf_vectorizer.joblib, label_encoder.joblib")

print("\\n📦 All files ready for website integration:")
print("  • model_comparison.csv — Model performance table")
print("  • per_class_metrics.csv — Per-category F1 by model")
print("  • analysis_results.json — Complete analysis data")
print("  • best_resume_model.joblib — Trained model")
print("  • tfidf_vectorizer.joblib — TF-IDF vectorizer")
print("  • label_encoder.joblib — Label encoder")"""))

# ═══════════════════════════════════════════════════════════════════════════════
# PART D — SUMMARY (Sections 23-25)
# ═══════════════════════════════════════════════════════════════════════════════

cells.append(md("""\
---
# Part D — Summary

## 23. Performance Discussion

**Best performing model**: Typically **LinearSVC** or **Logistic Regression** with
TF-IDF features achieves the highest F1 scores (often > 95%).

**Key findings from analysis**:
- Resume categories have distinct vocabulary patterns (Section 6-7)
- Some categories cluster closely (e.g., similar IT roles) — visible in similarity matrix (Section 8)
- Category sample sizes vary significantly — tier analysis shows imbalance (Section 3)
- Certain categories are inherently harder to classify due to vocabulary overlap (Section 17)

**Challenges**:
- Class imbalance affects minority category performance
- Overlapping categories (e.g., *Python Developer* vs *Data Science*) cause confusion
- Resume length varies greatly across categories, affecting feature extraction"""))

cells.append(md("""\
---
## 24. Future Improvements

1. **Transformer models** — Fine-tune BERT/RoBERTa for higher accuracy
2. **Named Entity Recognition** — Extract skills, companies, degrees automatically
3. **Skill matching** — Rank candidates against specific job descriptions
4. **Larger datasets** — Combine multiple resume datasets for robustness
5. **Multi-label classification** — Allow resumes to belong to multiple categories
6. **Resume similarity search** — Use embeddings to find similar candidates
7. **Active learning** — Focus labeling effort on uncertain samples"""))

cells.append(md("""\
---
## 25. Reproducibility & Versions"""))

cells.append(code("""\
import sklearn, gensim
print(f"Python      : {__import__('sys').version}")
print(f"NumPy       : {np.__version__}")
print(f"Pandas      : {pd.__version__}")
print(f"Scikit-learn: {sklearn.__version__}")
print(f"NLTK        : {nltk.__version__}")
print(f"Gensim      : {gensim.__version__}")
if HAS_TF:
    print(f"TensorFlow  : {tf.__version__}")
print(f"\\nRandom seed : {SEED}")
print("\\n✅ Notebook complete — all results exported for website integration.")"""))

cells.append(md("""\
---
*End of comprehensive analysis notebook.*"""))

# ═══════════════════════════════════════════════════════════════════════════════
# BUILD THE NOTEBOOK JSON
# ═══════════════════════════════════════════════════════════════════════════════
notebook = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "cells": cells
}

out_path = "resume_analysis_complete.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"✅ Notebook written to {out_path}")
print(f"   Total cells: {len(cells)}")
md_cells = sum(1 for c in cells if c["cell_type"] == "markdown")
code_cells = sum(1 for c in cells if c["cell_type"] == "code")
print(f"   Markdown: {md_cells} | Code: {code_cells}")
