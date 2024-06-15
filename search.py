from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd

def search_products(query, tfidf_matrix, vectorizer, df, pricerange=float('inf')):
    df['Selling Price'] = df['Selling Price'].astype(str)
    df['Selling Price'] = df['Selling Price'].str.replace('$', '').str.replace(',', '')

    df['Selling Price'] = pd.to_numeric(df['Selling Price'], errors='coerce')

    df_filtered = df[df['Selling Price'] <= pricerange]

    if df_filtered.empty:
        return "No products found within the specified price range."

    query_vec = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

    filtered_indices = df_filtered.index.tolist()
    filtered_similarities = cosine_similarities[filtered_indices]

    top_index_within_filtered = filtered_indices[np.argmax(filtered_similarities)]

    result = df.iloc[top_index_within_filtered]

    formatted_price = f"${result['Selling Price']:.2f}"
    response_text = (
        f"Top result for you:\n"
        f"{result['Product Name']} - {formatted_price} - {result['Image']}"
    )

    return response_text, result['Product Name'], result