#---PROJECT: FAKE NEWS DETECTION USING NATURAL LANGUAGE PROCESSING---

# Importing libraries
import kagglehub
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#--DATA COLLECTION--

# Dataset 1
path1 = kagglehub.dataset_download("mahdimashayekhi/fake-news-detection-dataset")
df1 = pd.read_csv(f"{path1}/fake_news_dataset.csv")

print(df1["label"].unique())
print(df1["label"].value_counts())

# checking column name
print("Dataset 1 columns:", df1.columns.tolist())
rename_map = {}
if "title" not in df1.columns:
    for possible in ["headline", "Headline", "Title"]:
        if possible in df1.columns:
            rename_map[possible] = "title"
if "text" not in df1.columns:
    for possible in ["content", "article", "Content", "Text", "body"]:
        if possible in df1.columns:
            rename_map[possible] = "text"
if rename_map:
    df1.rename(columns=rename_map, inplace=True)
    print("Renamed columns:", rename_map)

# Convert labels to 0 = Real, 1 = Fake
df1["label"] = df1["label"].map({"real": 0, "fake": 1}).astype(int)

# Dataset 2
path2 = kagglehub.dataset_download("clmentbisaillon/fake-and-real-news-dataset")

true_df = pd.read_csv(f"{path2}/True.csv")
fake_df = pd.read_csv(f"{path2}/Fake.csv")

# Add labels before concatenating
true_df["label"] = 0   # Real
fake_df["label"] = 1   # Fake

# Merge True and Fake
merged_df2 = pd.concat([true_df, fake_df], ignore_index=True)

# Merge both datasets into one
df = pd.concat([df1, merged_df2], ignore_index=True)

# checking how many articles are repeated 
overlap_count = df.duplicated(subset=["text"]).sum()
print("Number of overlapping/duplicate articles across datasets (by text):", overlap_count)

print(df.shape)
print(df["label"].value_counts(dropna=False))

# check class balance
print(df["label"].value_counts(normalize=True) * 100)

# checking if one class (fake/real) is much bigger than the other
fake_count = (df["label"] == 1).sum()
real_count = (df["label"] == 0).sum()
print("Fake articles:", fake_count)
print("Real articles:", real_count)
print("Fake to Real ratio:", round(fake_count / real_count, 2))

# Drop unwanted columns
df.drop(columns=['date', 'source', 'author', 'category', 'subject'],
        inplace=True, errors='ignore')

# Shuffles all the rows in the DataFrame
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Check the dataset
print(df.info())
print(df.head())
print(df.columns)

#--DATA PREPROCESSING--

# Check for missing values
print(df.isnull().sum())

# Convert the label column to integer type
df['label'] = df['label'].astype(int)

# Check the number of fake and real news articles
print(df['label'].value_counts())

# Fill missing values in the title and text columns with an empty string
df['title'] = df['title'].fillna('')
df['text'] = df['text'].fillna('')

# Combine title and text into a single column for NLP
df['content'] = (df['title'] + " ") * 2 + df['text']

# Remove records with empty content
df = df[df["content"].str.strip() != ""]

# Remove very short articles (less than 5 words)
df = df[df["content"].str.split().str.len() > 5]

# Remove duplicate news articles before splitting into train/test
# so the same article does not end up in both sets
df.drop_duplicates(subset=["content"], inplace=True)

# Reset Dataframe index after preprocessing
df.reset_index(drop=True, inplace=True)
print("Dataset Shape after preprocessing:", df.shape)

# Import libraries for text processing
import re
import string
import nltk

# Download NLTK resources
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# Import NLTK modules
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Create stopwords set and lemmatizer
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# Function to clean text
def word_drop(text):
    # Convert to lowercase
    text = text.lower()

    # expand common short forms 
    contractions = {
        "won't": "will not", "can't": "cannot", "n't": " not",
        "'re": " are", "'s": " is", "'d": " would",
        "'ll": " will", "'t": " not", "'ve": " have", "'m": " am"
    }
    for pattern, replacement in contractions.items():
        text = text.replace(pattern, replacement)

    # Remove text inside square brackets
    text = re.sub(r'\[.*?\]', '', text)

    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>+', '', text)

    # Remove punctuation
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)

    # Remove numbers
    text = re.sub(r'\w*\d\w*', '', text)

    # fix repeated letters 
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)

    # remove extra spaces left after cleaning
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenize the text
    words = word_tokenize(text)

    # Remove stopwords, remove very short leftover words, and apply lemmatization
    words = [lemmatizer.lemmatize(word) for word in words
             if word not in stop_words and len(word) > 2]

    # Join the words back into a sentence
    return " ".join(words)


# Apply the function
df['content'] = df['content'].apply(word_drop)

# Check the cleaned data
print(df[['content', 'label']].head())

#--FEATURE ENGINEERING--

# Features
X = df['content']
# Target
y = df['label']

# Train-Test Split (80% train, 20% test, stratify keeps fake/real ratio same in both)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Convert text into numerical features (TF-IDF)
from sklearn.feature_extraction.text import TfidfVectorizer
# Create TF-IDF object
tfidf = TfidfVectorizer(
    stop_words='english',
    lowercase=True,
    strip_accents='unicode',
    max_df=0.9,
    min_df=5,
    max_features=400000,
    ngram_range=(1, 3),
    sublinear_tf=True
)
# Fit and transform training data
X_train = tfidf.fit_transform(X_train)
# Transform testing data
X_test = tfidf.transform(X_test)

# Train Logistic Regression Model using GridSearchCV to pick best C value
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

print("\n--- GridSearchCV for Logistic Regression ---")
lr_params = {"C": [1, 5, 10]}
lr_grid = GridSearchCV(
    LogisticRegression(solver='liblinear', max_iter=3000, random_state=42, class_weight='balanced'),
    param_grid=lr_params,
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)
lr_grid.fit(X_train, y_train)

print("Best C value chosen:", lr_grid.best_params_["C"])
print("Best CV Accuracy:", lr_grid.best_score_)

# use the best model found
lr = lr_grid.best_estimator_

# Predict on test data
y_pred = lr.predict(X_test)

# Evaluate the model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Accuracy
print("Accuracy of Logistic Regression:", accuracy_score(y_test, y_pred))

# Confusion Matrix
print(confusion_matrix(y_test, y_pred))

# Classification Report
print(classification_report(y_test, y_pred))

# ROC-AUC score
from sklearn.metrics import roc_auc_score
print("ROC-AUC SCORE:", roc_auc_score(y_test, y_pred))

# checking train accuracy vs test accuracy to see if model is overfitting
print("Train Accuracy (Logistic Regression):", lr.score(X_train, y_train))
print("Test Accuracy (Logistic Regression):", lr.score(X_test, y_test))

# Visualize Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# import Naive Bayes model, tuned using GridSearchCV
from sklearn.naive_bayes import MultinomialNB

print("\n--- GridSearchCV for Naive Bayes ---")
nb_params = {"alpha": [0.01, 0.1, 1]}
nb_grid = GridSearchCV(
    MultinomialNB(),
    param_grid=nb_params,
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)
nb_grid.fit(X_train, y_train)

print("Best alpha value chosen:", nb_grid.best_params_["alpha"])
print("Best CV Accuracy:", nb_grid.best_score_)

# Create the model
nb = nb_grid.best_estimator_

# Predict on test data
y_pred_nb = nb.predict(X_test)

# Import evaluation metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Print accuracy
print("Accuracy of Naive bayes:", accuracy_score(y_test, y_pred_nb))

# Print confusion matrix
cm = confusion_matrix(y_test, y_pred_nb)
print(cm)

# Print classification report
print(classification_report(y_test, y_pred_nb))

print("Train Accuracy (Naive Bayes):", nb.score(X_train, y_train))
print("Test Accuracy (Naive Bayes):", nb.score(X_test, y_test))

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Naive Bayes - Confusion Matrix")
plt.show()

# Import Passive Aggressive Classifier, tuned using GridSearchCV
from sklearn.linear_model import PassiveAggressiveClassifier

print("\n--- GridSearchCV for Passive Aggressive Classifier ---")
pac_params = {"C": [0.1, 0.5, 1]}
pac_grid = GridSearchCV(
    PassiveAggressiveClassifier(max_iter=2000, random_state=42, class_weight='balanced'),
    param_grid=pac_params,
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)
pac_grid.fit(X_train, y_train)

print("Best C value chosen:", pac_grid.best_params_["C"])
print("Best CV Accuracy:", pac_grid.best_score_)

# Create the model
pac = pac_grid.best_estimator_

# Predict on test data
y_pred_pac = pac.predict(X_test)

# Import evaluation metrics
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Print accuracy
print("Accuracy of Passive Aggressive Classifier:", accuracy_score(y_test, y_pred_pac))

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred_pac)

# Print confusion matrix
print(cm)

# Print classification report
print(classification_report(y_test, y_pred_pac))

print("Train Accuracy (Passive Aggressive):", pac.score(X_train, y_train))
print("Test Accuracy (Passive Aggressive):", pac.score(X_test, y_test))

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Passive Aggressive Classifier - Confusion Matrix")
plt.show()

# Compare the models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

lr_acc = accuracy_score(y_test, y_pred)
nb_acc = accuracy_score(y_test, y_pred_nb)
pac_acc = accuracy_score(y_test, y_pred_pac)

lr_pre = precision_score(y_test, y_pred)
nb_pre = precision_score(y_test, y_pred_nb)
pac_pre = precision_score(y_test, y_pred_pac)

lr_rec = recall_score(y_test, y_pred)
nb_rec = recall_score(y_test, y_pred_nb)
pac_rec = recall_score(y_test, y_pred_pac)

lr_f1 = f1_score(y_test, y_pred)
nb_f1 = f1_score(y_test, y_pred_nb)
pac_f1 = f1_score(y_test, y_pred_pac)

# Create comparison table
comparison = pd.DataFrame({
    "Algorithm": [
        "Logistic Regression",
        "Naive Bayes",
        "Passive Aggressive"
    ],
    "Accuracy": [lr_acc, nb_acc, pac_acc],
    "Precision": [lr_pre, nb_pre, pac_pre],
    "Recall": [lr_rec, nb_rec, pac_rec],
    "F1 Score": [lr_f1, nb_f1, pac_f1]
})

# Display comparison table
print(comparison)

# Select the best model
models = {
    "Logistic Regression": lr_acc,
    "Naive Bayes": nb_acc,
    "Passive Aggressive": pac_acc
}
best_model = max(models, key=models.get)
print("Best Model:", best_model)
print("Best Accuracy:", models[best_model])

# Select the best trained model
if best_model == "Logistic Regression":
    final_model = lr
elif best_model == "Naive Bayes":
    final_model = nb
else:
    final_model = pac

# Save the model
import joblib

joblib.dump(final_model, "fake_news_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")

print("Model and TF-IDF Vectorizer saved successfully.")

# Bar chart for Accuracy
plt.figure(figsize=(8, 5))
plt.bar(comparison["Algorithm"], comparison["Accuracy"], color="skyblue")
plt.title("Accuracy Comparison")
plt.xlabel("Algorithms")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.show()

# Bar chart for Precision
plt.figure(figsize=(8, 5))
plt.bar(comparison["Algorithm"], comparison["Precision"], color="lightgreen")
plt.title("Precision Comparison")
plt.xlabel("Algorithms")
plt.ylabel("Precision")
plt.ylim(0, 1)
plt.show()

# Bar chart for Recall
plt.figure(figsize=(8, 5))
plt.bar(comparison["Algorithm"], comparison["Recall"], color="salmon")
plt.title("Recall Comparison")
plt.xlabel("Algorithms")
plt.ylabel("Recall")
plt.ylim(0, 1)
plt.show()

# Bar chart for F1 Score
plt.figure(figsize=(8, 5))
plt.bar(comparison["Algorithm"], comparison["F1 Score"], color="orange")
plt.title("F1 Score Comparison")
plt.xlabel("Algorithms")
plt.ylabel("F1 Score")
plt.ylim(0, 1)
plt.show()

# Combined chart with all metrics together
comparison_plot = comparison.set_index("Algorithm")
comparison_plot.plot(kind="bar", figsize=(10, 6))
plt.title("Comparison of Machine Learning Models")
plt.xlabel("Models")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.grid(axis="y")
plt.show()

#-- PROJECT SUMMARY--
print("\n========== PROJECT SUMMARY ==========")
print("Project Title: Fake News Detection Using NLP")
print("Total Records:", len(df))
print("Total Features (TF-IDF):", X_train.shape[1])
print("Training Samples:", X_train.shape[0])
print("Testing Samples:", X_test.shape[0])
print("Models Trained:", 3)
print("Best Model:", best_model)
print("Best Accuracy:", round(models[best_model], 4))
print("Model Saved: fake_news_model.pkl")
print("TF-IDF Saved: tfidf_vectorizer.pkl")
print("Prediction Labels: 0 = Real News, 1 = Fake News")
