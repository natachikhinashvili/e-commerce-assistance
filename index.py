from flask import Flask, request, jsonify, render_template
from search import search_products
from processdata import tfidf_matrix, vectorizer, df

app = Flask(__name__)


@app.route('/', methods=['GET'])
def main():
    return render_template('index.html')


@app.route('/get', methods=['GET', 'POST'])
def search():    
    user_input = request.args.get('msg')
    app.logger.info(f"Received user input: {user_input}")
    result = search_products(user_input, tfidf_matrix, vectorizer, df)

    response_text = f"Top result for you :\n{result['Product Name']} - {result['Selling Price']} - {result['Product Url']}"

    app.logger.info(f"Response: {response_text}")
    return jsonify(response=response_text)


if __name__ == "__main__":
    app.run(debug=True)


