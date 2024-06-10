from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from processdata import tfidf_matrix, vectorizer, df

def search_products(query, tfidf_matrix, vectorizer, df):
    query_vec = vectorizer.transform([query])

    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

    top_indices = np.argsort(cosine_similarities)[-10:][::-1]

    results = df.iloc[top_indices]
    return results

user_query = "red dress"

results = search_products(user_query, tfidf_matrix, vectorizer, df)

print(results)
