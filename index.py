from flask import Flask, request, jsonify, render_template
from search import search_products
from processdata import tfidf_matrix, vectorizer, df
from detectprice import detect_price_request
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

app = Flask(__name__)
lemmatizer = WordNetLemmatizer()

@app.route('/', methods=['GET'])
def main():
    return render_template('index.html')


@app.route('/get', methods=['GET', 'POST'])
def search():    
    user_input = request.args.get('msg')

    app.logger.info(f"Received user input: {user_input}")
    
    tokenized =  word_tokenize(user_input)

    lemmatized =  [lemmatizer.lemmatize(word) for word in tokenized]

    transformed = ' '.join(lemmatized)

    price_request, price = detect_price_request(transformed)
    if price_request:
        result = search_products(transformed, tfidf_matrix, vectorizer, df, price)
    else:
        result = search_products(transformed, tfidf_matrix, vectorizer, df )

    return jsonify(response=result)


if __name__ == "__main__":
    app.run(debug=True)


