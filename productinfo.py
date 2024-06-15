from processdata import df
from generatesentence import generate_sentence

def productinfo(product_name):
    product = df['Product Name'].str.lower() == product_name.lower()
    if not product.empty:
        index = product.idxmax()
        details = df.at[index,'About Product']
        producturl = df.at[index,'Product Url']
        trimmed = details.replace("Make sure this fits by entering your model number.", '').strip()
        words = trimmed.split('|')
        generated_sentence = generate_sentence(words)
        response = (
                    f"{generated_sentence}\n"
                    f"You can get it at the url: {producturl}"
                )
        
        return response
    else:
        return f"Product '{product_name}' not found."
