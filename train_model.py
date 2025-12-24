
# train_model.py (robust + debug version)
import pandas as pd
import re
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib
import numpy as np

# ---------- feature extraction ----------
SUSPICIOUS_WORDS = [
    "login", "verify", "update", "secure", "account",
    "bank", "free", "confirm", "payment", "alert",
    "winner", "prize", "gift", "reward"
]

def extract_features(url: str):
    parsed = urlparse(url)
    features = {
        "url_length": len(url),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_slashes": url.count("/"),
        "has_at": 1 if "@" in url else 0,
        "has_ip": 1 if re.match(r"^(http[s]?://)?(\d{1,3}\.){3}\d{1,3}", url) else 0,
        "https": 1 if parsed.scheme == "https" else 0,
        "domain_length": len(parsed.netloc),
        "suspicious_words_count": sum(1 for w in SUSPICIOUS_WORDS if w in url.lower())
    }
    return features

def build_feature_dataframe(df: pd.DataFrame):
    rows = [extract_features(u) for u in df["url"]]
    return pd.DataFrame(rows)

# ---------- robust loading & cleaning ----------
def load_and_clean(path="url.csv"):
    print(f"Loading dataset from \"{path}\" ...")
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    print("Raw shape:", df.shape)
    print("Columns:", list(df.columns))

    df.columns = [c.strip().lower() for c in df.columns]

    if "url" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain columns: url, label")

    df["url"] = df["url"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    print("\nPreview (first 8 rows):")
    print(df.head(8).to_string(index=False))

    unique_labels = sorted(df["label"].unique())
    print("\nUnique raw label values (pre-clean):", unique_labels)

    mapping = {
        "phishing": "1", "malicious": "1", "bad": "1", "spam": "1",
        "phish": "1",
        "legitimate": "0", "benign": "0", "good": "0", "safe": "0"
    }
    df["label"] = df["label"].replace(mapping)

    df = df[df["url"].str.strip() != ""]
    before = df.shape[0]
    df = df[df["label"].isin(["0", "1"])]
    removed = before - df.shape[0]
    print(f"\nRows removed because label not in {{0,1}}: {removed}")

    if df.empty:
        raise ValueError("No valid rows with label 0/1 found after cleaning.")

    df["label"] = df["label"].astype(int)

    print("\nLabel distribution after cleaning:")
    print(df["label"].value_counts())

    return df

# ---------- training ----------
def main():
    df = load_and_clean("url.csv")
    X = build_feature_dataframe(df)
    y = np.asarray(df["label"]).ravel()

    print("\nFeature matrix shape:", X.shape)
    print("Unique labels in y:", np.unique(y))

    if len(np.unique(y)) < 2:
        raise ValueError("Dataset does not contain both classes 0 AND 1.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\nTraining samples:", X_train.shape[0])
    print("Test samples:", X_test.shape[0])

    print("\nTraining model ...")
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    print("Training completed.")

    y_pred = model.predict(X_test)

    y_pred = np.asarray(y_pred).ravel()

    print("Unique predicted labels:", np.unique(y_pred))

    acc = accuracy_score(y_test, y_pred)
    print("\nAccuracy:", acc)
    print("\nClassification report:\n")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, "phishing_model.pkl")
    joblib.dump(list(X.columns), "feature_columns.pkl")

    print("\nSaved model: phishing_model.pkl")
    print("Saved features: feature_columns.pkl")

if __name__ == "__main__":
    main()
