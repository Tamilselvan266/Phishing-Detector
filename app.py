import tkinter as tk
from tkinter import Canvas, messagebox
import joblib
import pandas as pd
from train_model import extract_features

# ------------------- Load Model -------------------
model = joblib.load("phishing_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

def prepare_features(url):
    feats = extract_features(url)
    df = pd.DataFrame([feats])
    df = df[feature_columns]
    return df

# ------------------- GUI Functions -------------------
def detect_url():
    url = url_entry.get().strip()
    if not url:
        messagebox.showwarning("Missing URL", "Please enter a URL.")
        return

    X = prepare_features(url)
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][pred]

    # Animate the result
    result_label.place(x=50, y=280)
    slide_in(result_label)

    if pred == 1:
        result_label.config(
            text=f"⚠️  PHISHING Detected!\nConfidence: {prob*100:.2f}%",
            fg="#FF3B3B"
        )
    else:
        result_label.config(
            text=f"✅  Legitimate URL\nConfidence: {prob*100:.2f}%",
            fg="#00B368"
        )


def slide_in(widget):
    """Smooth slide-down animation."""
    for y in range(0, 20):
        widget.place_configure(y=260 + y)
        widget.update()
        widget.after(5)


def fade_in_text(label, text, delay=40):
    """Fade-in animation for the title."""
    label.config(text="")
    for i in range(len(text)):
        label.config(text=text[:i+1])
        label.update()
        label.after(delay)

# ------------------- UI Setup -------------------
app = tk.Tk()
app.title("Phishing URL Detector")
app.geometry("520x400")
app.resizable(False, False)

# Gradient Background
gradient = Canvas(app, width=520, height=400)
gradient.pack(fill="both", expand=True)

# Draw gradient
for i in range(400):
    r = int(240 - (i * 0.15))
    g = int(245 - (i * 0.2))
    b = 255
    color = f"#{r:02x}{g:02x}{b:02x}"
    gradient.create_line(0, i, 520, i, fill=color)

# Title
title_label = tk.Label(
    app, text="", font=("Segoe UI", 20, "bold"),
    bg="#F0F5FF", fg="#2A4D85"
)
title_label.place(x=120, y=20)

# Fade-in animation
fade_in_text(title_label, "Phishing URL Detector")

# URL Input
url_entry = tk.Entry(app, font=("Segoe UI", 12), width=40, bd=2, relief="flat")
url_entry.place(x=70, y=120)

# Detect Button (Animated)
def on_enter(e):
    detect_btn.config(bg="#4CAF50", fg="white", width=16)

def on_leave(e):
    detect_btn.config(bg="#3B7A3A", fg="white", width=14)

detect_btn = tk.Button(
    app, text="Detect URL", font=("Segoe UI", 12, "bold"),
    bg="#3B7A3A", fg="white", width=14, height=1,
    relief="flat", command=detect_url
)
detect_btn.place(x=190, y=175)

detect_btn.bind("<Enter>", on_enter)
detect_btn.bind("<Leave>", on_leave)

# Result Label
result_label = tk.Label(
    app, text="", font=("Segoe UI", 14, "bold"),
    bg="#F0F5FF"
)

app.mainloop()
