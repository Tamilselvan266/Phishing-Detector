import joblib
import pandas as pd
from train_model import extract_features

def load_model():
    model = joblib.load("phishing_model.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
    return model, feature_columns

def prepare_features(url, feature_columns):
    features = extract_features(url)
    df = pd.DataFrame([features])
    df = df[feature_columns]   # ensure column order matches training
    return df

def main():
    print("\n🔍 Phishing URL Detection Tool Ready!\n")

    model, feature_columns = load_model()

    while True:
        url = input("Enter a URL (or 'q' to quit): ").strip()

        if url.lower() == "q":
            print("Exiting tool. Goodbye!")
            break

        X = prepare_features(url, feature_columns)
        pred = model.predict(X)[0]
        prob = model.predict_proba(X)[0][pred]

        if pred == 1:
            print(f"⚠️  PHISHING detected!  (Confidence: {prob*100:.2f}%)\n")
        else:
            print(f"✅ Legitimate URL. (Confidence: {prob*100:.2f}%)\n")


if __name__ == "__main__":
    main()
