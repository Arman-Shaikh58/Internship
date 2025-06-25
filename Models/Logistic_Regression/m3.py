from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# 1. Sample Training Data
texts = [
    "I hate you and your people",       # hate
    "Go away, no one wants you here",   # hate
    "You're such an idiot",             # offensive
    "You are amazing and kind",         # non-hate
    "Have a beautiful day",             # non-hate
]

labels = [1, 1, 1, 0, 0]  # 1 = hate/offensive, 0 = non-hate

# 2. Create Model Pipeline (TF-IDF + Logistic Regression)
model = make_pipeline(TfidfVectorizer(), LogisticRegression())

# 3. Train the Model
model.fit(texts, labels)

# 4. Test on Unseen New Text
new_texts = [
    "People like you disgust me",        # should be hate
    "Let's all be friends and love",     # should be non-hate
    "You're the worst human ever",       # offensive/hate
]

# 5. Predict
preds = model.predict(new_texts)
probs = model.predict_proba(new_texts)

# 6. Show Results
for text, pred, prob in zip(new_texts, preds, probs):
    label = "Hate/Offensive" if pred == 1 else "Non-Hate"
    confidence = round(prob[pred] * 100, 2)
    print(f"'{text}' → {label} ({confidence}%)")
