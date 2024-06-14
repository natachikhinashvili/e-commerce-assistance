import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from fakedata import data
import re

nlp = spacy.load("en_core_web_sm")

X = []
y = []
for text, label in data:
    doc = nlp(text)
    tokens = [token.text for token in doc]
    X.append(" ".join(tokens))
    y.append(label)

vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

classifier = MultinomialNB()
classifier.fit(X_vectorized, y)

def detect_price_request(user_input):
    price_regex = r"\$?(\d+(\.\d+)?)"
    matches = re.findall(price_regex, user_input)
    if matches:
        price = float(matches[0][0])  
        doc = nlp(user_input)
        tokens = [token.text for token in doc]
        user_input_vectorized = vectorizer.transform([" ".join(tokens)])
        prediction = classifier.predict(user_input_vectorized)
        return prediction[0] == True, price
    else:
        return False, None