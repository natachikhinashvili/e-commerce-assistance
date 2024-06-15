from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

def generate_sentence(words):
    input_text = ' '.join(words)
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    
    output = model.generate(input_ids, max_length=50, num_return_sequences=1, 
                            pad_token_id=tokenizer.eos_token_id)
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text.split('\n')[0].strip() + '.'