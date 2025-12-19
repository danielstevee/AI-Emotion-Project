import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB


nltk.download('stopwords')
nltk.download('wordnet')


train = pd.read_csv("training.csv")
test  = pd.read_csv("test.csv")

X_train_raw = train['text']
y_train = train['label']

X_test_raw = test['text']
y_test = test['label']


stop_words = set(stopwords.words('english'))
lem = WordNetLemmatizer()

def clean_txt(text):
    text = text.lower()
    text = re.sub(r"http\S+|[^a-z\s]", "", text)
    text = " ".join(lem.lemmatize(w) for w in text.split() if w not in stop_words)
    return text

X_train = X_train_raw.apply(clean_txt)
X_test = X_test_raw.apply(clean_txt)


tfidf = TfidfVectorizer()
X_train_tf = tfidf.fit_transform(X_train)
X_test_tf = tfidf.transform(X_test)


models = {
    "SVM": SVC(kernel='rbf', C=100, gamma=0.01),
    "KNN": KNeighborsClassifier(n_neighbors=11, weights='distance', metric='euclidean'),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_split=2, min_samples_leaf=1),
    "Logistic Regression": LogisticRegression(C=1, penalty='l2', solver='liblinear', max_iter=1000),
    "Multinomial NB": MultinomialNB(alpha=0.1)
}

results = {}   

for name, model in models.items():
    model.fit(X_train_tf, y_train)
    preds = model.predict(X_test_tf)

    acc = accuracy_score(y_test, preds)
    results[name] = {"accuracy": acc}

    print(f"\n===== {name} =====")
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, zero_division=0))

best_model = max(results, key=lambda x: results[x]['accuracy'])
best_accuracy = results[best_model]['accuracy']

print("\n========================")
print(" BEST MODEL RESULT")
print("========================")
print(f"Best Model     : {best_model}")
print(f"Accuracy       : {best_accuracy:.4f}")