import spacy
from processdata import df
from productinfo import productinfo

nlp = spacy.load('en_core_web_sm')

def aboutproduct(query, product, mydf):
    doc = nlp(query)
    question_words = {'what', 'who', 'when', 'where', 'why', 'how', 'tell', 'about'}
    if any(token.lemma_.lower() in question_words for token in doc):
        return productinfo(product)
    return None