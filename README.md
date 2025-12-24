# Phishing-Detector

## Project Overview
Phishing websites trick users into entering sensitive information such as usernames, passwords, and bank details. This project implements a **Machine Learning–based Phishing Website Detection Tool** that analyzes URLs and classifies them as **Phishing** or **Legitimate**.

The application is developed for **cybersecurity learning and internship purposes** and demonstrates how real-world phishing detection systems work.


## Objectives
- To detect phishing websites using URL-based features
- To apply Machine Learning for cybersecurity problems
- To help users identify malicious URLs
- To provide both **CLI** and **GUI (Tkinter)** based interaction


## 🛠 Technologies Used
- **Python** – Core programming language
- **Pandas** – Data loading and preprocessing
- **Regex (re module)** – URL pattern analysis
- **Scikit-learn** – Machine Learning model training
- **Tkinter** – GUI application (optional)
- **Joblib** – Model saving and loading


## Dataset Description
The dataset consists of URLs labeled as:
- **1 → Phishing**
- **0 → Legitimate**

### Dataset Summary:
- Total URLs: 118  
- Phishing URLs: 51  
- Legitimate URLs: 67  

The dataset is stored in `url.csv` and used for training and testing the ML model.


##  Machine Learning Approach
- Feature extraction from URLs (length, dots, hyphens, HTTPS, IP address, suspicious words, etc.)
- Algorithm used: **Random Forest Classifier**
- Train–test split: 80% training, 20% testing
- Evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1-score


##  Features
- Detects phishing and legitimate URLs
- Machine Learning–based classification
- Confidence score for predictions
- Command-line prediction tool
- Tkinter-based GUI with attractive UI and animations
- Offline detection (no internet required)


## 🖥 Application Flow
1. User enters a URL
2. URL features are extracted
3. Trained ML model predicts the result
4. Output is displayed as:
   -  Phishing
   -  Legitimate


##  Project Structure
phishing-detector/
│
├── train_model.py # Model training script
├── predict_url.py # CLI phishing detector
├── gui_app.py # Tkinter GUI application
├── url.csv # Dataset (phishing & legitimate URLs)
├── phishing_model.pkl # Trained ML model
├── feature_columns.pkl # Feature order for prediction
├── screenshots/ # Application screenshots
├── README.md # Project documentation
└── venv/ # Virtual environment