# Spam-detector
A simple machine learning project to detect spam messages using Python and scikit-learn.
import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, precision_recall_curve
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
import numpy as np

# -----------------------
# Helper text cleaning
# -----------------------
def clean_text(s):
    s = s.lower()
    s = re.sub(r"http\S+", " ", s)               # remove urls
    s = re.sub(r"[^a-z0-9\s]", " ", s)           # keep alphanumeric and spaces
    s = re.sub(r"\s+", " ", s).strip()           # collapse whitespace
    return s

# -----------------------
# Load dataset
# -----------------------
# Expecting a CSV with columns: 'label' and 'text'
# where label is 'spam' or 'ham' (or 1 and 0)
if os.path.exists("spam.csv"):
    df = pd.read_csv("spam.csv")
    # try common variations
    if 'v1' in df.columns and 'v2' in df.columns:
        df = df.rename(columns={'v1':'label', 'v2':'text'})
    elif 'label' not in df.columns or 'text' not in df.columns:
        raise RuntimeError("spam.csv found but expected columns 'label' and 'text'.")
    df = df[['label','text']].dropna()
    # normalize labels to 0/1
    df['label'] = df['label'].map(lambda x: 1 if str(x).strip().lower() in ('spam','1','true','t','yes') else 0)
else:
    # Small demo dataset - fine for quick testing and learning
    sample_texts = [
        ("ham","Hey, are we still meeting for lunch today?"),
        ("spam","Congratulations! You've won a $1000 gift card. Click here to claim."),
        ("ham","Don't forget to bring the documents."),
        ("spam","Get cheap meds online, no prescription needed."),
        ("ham","Can you send me the assignment file?"),
        ("spam","You have been selected for a cash prize! Reply YES"),
        ("ham","I'll call you in 10 minutes."),
        ("spam","Earn money from home—no skills required!")
    ]
    df = pd.DataFrame(sample_texts, columns=['label','text'])
    df['label'] = df['label'].map(lambda x: 1 if x=='spam' else 0)

print("Dataset size:", len(df))
print(df.label.value_counts())

# -----------------------
# Preprocess texts
# -----------------------
df['text_clean'] = df['text'].astype(str).apply(clean_text)

X = df['text_clean'].values
y = df['label'].values

# -----------------------
# Train / Test split
# -----------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------
# Vectorizer and model
# -----------------------
tfidf = TfidfVectorizer(ngram_range=(1,2), min_df=1, stop_words='english')
# LogisticRegression with class_weight to help imbalance
clf = LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear')

# Transform training data
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf  = tfidf.transform(X_test)

# -----------------------
# Optional: SMOTE oversampling (uncomment to use)
# -----------------------
USE_SMOTE = True  # set True to enable SMOTE
if USE_SMOTE:
    print("Applying SMOTE to training data...")
    sm = SMOTE(random_state=42)
    X_train_tfidf, y_train = sm.fit_resample(X_train_tfidf, y_train)

# -----------------------
# Train
# -----------------------
clf.fit(X_train_tfidf, y_train)

# -----------------------
# Evaluate
# -----------------------
y_pred = clf.predict(X_test_tfidf)
y_prob = clf.predict_proba(X_test_tfidf)[:,1] if hasattr(clf, "predict_proba") else clf.decision_function(X_test_tfidf)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['ham','spam']))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['ham','spam'])
disp.plot()
plt.title("Confusion Matrix")
plt.show()

# ROC AUC (if both classes present)
if len(np.unique(y_test)) == 2:
    try:
        roc = roc_auc_score(y_test, y_prob)
        print("ROC AUC:", round(roc, 4))
    except Exception as e:
        print("ROC AUC calculation failed:", e)

# Precision-Recall curve
if len(np.unique(y_test)) == 2:
    precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curve")
    plt.show()

# -----------------------
# Quick demo: predict custom messages
# -----------------------
def predict_message(msg):
    c = clean_text(msg)
    v = tfidf.transform([c])
    p = clf.predict(v)[0]
    prob = clf.predict_proba(v)[0,1] if hasattr(clf, "predict_proba") else None
    label = "SPAM" if p==1 else "HAM (not spam)"
    return label, prob

print("\nTry sample predictions:")
tests = [
    "WIN a brand new iPhone! Click the link to claim.",
    "Are you coming to the meeting tomorrow?",
    "Lowest price on car insurance. Visit our site now!"
]
for t in tests:
    lab, pr = predict_message(t)
    print(f"Text: {t}\n-> {lab} (prob={pr})\n")

