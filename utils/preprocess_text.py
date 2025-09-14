from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def preprocess_text(text, max_length=128):
    inputs = tokenizer(text, padding='max_length', truncation=True, max_length=max_length, return_tensors="pt")
    return inputs['input_ids'], inputs['attention_mask']
