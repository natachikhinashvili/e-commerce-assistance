from flask import Flask, request, jsonify
import spacy
from search import search_products
from processdata import tfidf_matrix, vectorizer, df

app = Flask(__name__)

nlp = spacy.load("en_core_web_sm")
user = input('How can i help you?')

result = nlp(user)

@app.route('/', methods=['GET'])
def search():
    results = search_products(user, tfidf_matrix, vectorizer, df)
    return jsonify(results.to_dict(orient='records'))

if __name__ == "__main__":
    app.run(debug=True)


