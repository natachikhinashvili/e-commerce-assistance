import pandas as pd
import math
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv('./sample_data_10k.csv')


df['combined_text'] = df[['Product Name', 
                          'Category', 
                          'About Product', 
                          'Product Specification', 
                          'Technical Details',
                          'Selling Price',
                          'Model Number',
                          ]].fillna('').agg(' '.join, axis=1)


vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(df['combined_text'])

import pickle
with open('tfidf_matrix.pkl', 'wb') as f:
    pickle.dump(tfidf_matrix, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
with open('df.pkl', 'wb') as f:
    pickle.dump(df, f)