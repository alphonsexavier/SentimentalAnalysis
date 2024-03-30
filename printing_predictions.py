import tensorflow as tf
from transformers import BertTokenizer

# Load the saved model
loaded_model = tf.saved_model.load("C:/Users/Alphy/Documents/Clark/Sem 2/MSCS 3027 - Social Informatics/Designing a model/Outputs/bert_model/saved_model/1/")

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def preprocess(X):
    import re
    def text_clean(text):
        temp = text.lower()
        temp = re.sub("@[A-Za-z0-9_]+","", temp)
        temp = re.sub("#[A-Za-z0-9_]+","", temp)
        temp = re.sub(r"http\S+", "", temp)
        temp = re.sub(r"www.\S+", "", temp)
        temp = re.sub("[0-9]","", temp)
        return temp
    X_cleaned = [text_clean(text) for text in X]
    return X_cleaned

def predict_text(text):
    # Preprocess the text
    cleaned_text = preprocess([text])
    encoded_text = tokenizer(cleaned_text[0], return_tensors="tf", max_length=128, padding=True, truncation=True)

    # Get the concrete function for inference
    infer = loaded_model.signatures["serving_default"]

    # Make predictions
    predictions = infer(input_ids=encoded_text.input_ids, attention_mask=encoded_text.attention_mask, token_type_ids=encoded_text.token_type_ids)

    return predictions

# Example usage
sample_text = "What the hell is wrong with you"
predictions = predict_text(sample_text)

# Assuming predictions is your dictionary containing the logits
logits = predictions['logits']
probabilities = tf.nn.softmax(logits)

# Assuming probabilities[:, 1] corresponds to the probability of the positive class
positive_probability = probabilities[:, 1].numpy()[0]

# Define the threshold
threshold = 0.5

# Classify the text based on the threshold
if positive_probability >= threshold:
    print("The sample text is predicted to be positive.")
else:
    print("The sample text is predicted to be negative.")