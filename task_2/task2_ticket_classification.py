# ===================== IMPORTS =====================
import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ===================== LOAD DATA =====================
file_path = "data/raw/all_tickets_processed_improved_v3.csv"
df = pd.read_csv(file_path)

print("Dataset loaded successfully")
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns)


# ===================== SELECT REQUIRED COLUMNS =====================
df = df[['Document', 'Topic_group']]
df.dropna(inplace=True)

print("\nData preview:")
print(df.head())


# ===================== TEXT PREPROCESSING =====================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_text'] = df['Document'].apply(clean_text)

# ===================== PRIORITY LOGIC =====================
def assign_priority(text):
    text = text.lower()

    high_keywords = [
        'urgent', 'immediately', 'down', 'outage', 'failed',
        'crash', 'not working', 'error', 'unable to access'
    ]

    medium_keywords = [
        'request', 'issue', 'problem', 'unable', 'delay'
    ]

    low_keywords = [
        'how to', 'information', 'query', 'clarification', 'guidance'
    ]

    for word in high_keywords:
        if word in text:
            return "High"

    for word in medium_keywords:
        if word in text:
            return "Medium"

    for word in low_keywords:
        if word in text:
            return "Low"

    return "Medium"   # default


# Remove very short texts
df = df[df['clean_text'].str.len() > 20]

print("\nCleaned text sample:")
print(df[['Document', 'clean_text']].head())


# ===================== LABEL ENCODING =====================
label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['Topic_group'])

print("\nLabel classes:")
print(list(label_encoder.classes_))


# ===================== TRAIN-TEST SPLIT =====================
X = df['clean_text']
y = df['label_encoded']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ===================== TF-IDF VECTORIZATION =====================
tfidf = TfidfVectorizer(
    max_features=15000,
    ngram_range=(1, 2),
    stop_words='english',
    min_df=3
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("\nTF-IDF matrix shape:", X_train_tfidf.shape)


# ===================== LOGISTIC REGRESSION MODEL =====================
model = LogisticRegression(
    max_iter=2000,
    class_weight='balanced',
    n_jobs=-1
)

model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)


# ===================== EVALUATION =====================
accuracy = accuracy_score(y_test, y_pred)
print("\n🎯 Accuracy:", accuracy)

print("\n📊 Classification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_
))

print("\n🧮 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# ===================== APPLY PRIORITY LOGIC =====================
df['Predicted_Priority'] = df['clean_text'].apply(assign_priority)

print("\nPriority distribution:")
print(df['Predicted_Priority'].value_counts())




# ===================== SAMPLE PREDICTION =====================
def predict_ticket(text):
    cleaned = clean_text(text)
    vec = tfidf.transform([cleaned])
    pred = model.predict(vec)[0]
    return label_encoder.inverse_transform([pred])[0]

print("\n🔍 Sample Prediction:")
sample = "VPN connection fails after password reset"
print("Ticket:", sample)
print("Predicted Category:", predict_ticket(sample))


print("\n✅ Task 2 – IT Ticket Classification Completed Successfully")



# ===================== FINAL DECISION OUTPUT =====================
def classify_ticket_with_priority(ticket_text):
    cleaned = clean_text(ticket_text)
    vec = tfidf.transform([cleaned])
    category_pred = model.predict(vec)[0]
    category = label_encoder.inverse_transform([category_pred])[0]

    priority = assign_priority(ticket_text)

    return category, priority


print("\n🔎 FINAL SYSTEM DEMO")
sample_ticket = "VPN is down and users cannot access the network urgently"
category, priority = classify_ticket_with_priority(sample_ticket)

print("Ticket:", sample_ticket)
print("Predicted Category:", category)
print("Assigned Priority:", priority)


# ===================== DYNAMIC USER INPUT =====================
print("\n🖥️  INTERACTIVE TICKET CLASSIFICATION SYSTEM")
print("Type 'exit' to stop\n")

while True:
    user_input = input("Enter support ticket text: ")

    if user_input.lower() == "exit":
        print("Exiting system. Goodbye!")
        break

    cleaned = clean_text(user_input)
    vec = tfidf.transform([cleaned])

    category_pred = model.predict(vec)[0]
    category = label_encoder.inverse_transform([category_pred])[0]

    priority = assign_priority(user_input)

    print("\n🔎 Prediction Result")
    print("Predicted Category:", category)
    print("Assigned Priority:", priority)

    print("-" * 50)
