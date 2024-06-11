from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def search_products(query, tfidf_matrix, vectorizer, df):
    query_vec = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

    top_index = np.argmax(cosine_similarities)

    result = df.iloc[top_index]
    return result
