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