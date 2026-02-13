from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, SimpleRNN, GRU, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, request, jsonify
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import TFGPT2LMHeadModel
import matplotlib.pyplot as plt
import os
from tensorflow.keras.layers import Input

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress INFO and WARNING logs


app = Flask(__name__)

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')  # Serve the index.html file


# Step 1: Load your dataset
# Replace 'your_dataset.txt' with your actual dataset file path
with open('dataset.txt', 'r', encoding='utf-8') as file:
    text = file.read().lower()

# Step 2: Prepare the dataset
# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])
total_words = len(tokenizer.word_index) + 1


# Create input-output sequences
input_sequences = []
for line in text.split('\n'):
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        input_sequences.append(n_gram_sequence)

# Padding sequences
max_sequence_length = max(len(x) for x in input_sequences)
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_length, padding='pre')

# Transformer-based GPT-2 Model
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")


# Step 6: Next words prediction
def predict_next_words(model, tokenizer, input_text, num_words=5):
    input_text = input_text.lower()
    generated_words = []
    
    # Generate words iteratively
    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([input_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_length-1, padding='pre')
        predicted = model.predict(token_list, verbose=0)
        
        # Get the index of the predicted word
        predicted_word_index = np.argmax(predicted)
        
        # Find the word corresponding to the predicted index
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted_word_index:
                output_word = word
                break
        
        # Append predicted word and update input_text for next prediction
        if output_word:
            generated_words.append(output_word)
            input_text += " " + output_word
        else:
            break
    
    return " ".join(generated_words)

def predict_gpt2(input_text, num_words):
    input_ids = gpt2_tokenizer.encode(input_text, return_tensors="pt")
    output = gpt2_model.generate(input_ids, max_length=len(input_ids[0]) + num_words, num_return_sequences=1)
    return gpt2_tokenizer.decode(output[0], skip_special_tokens=True)



@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_text = data.get("input_text", "")
    num_words = data.get("num_words", 5)  # Default to 5 words
    model_type = data.get("model_type", "lstm").lower()

    if not input_text:
        return jsonify({"error": "Input text is required."}), 400

    try:
        # Select the model based on the model_type
        if model_type == "lstm":
            model = tf.keras.models.load_model("LSTM_model.h5")
        elif model_type == "bi_lstm":
            model = tf.keras.models.load_model("Bi-LSTM_model.h5")
        elif model_type == "gru":
            model = tf.keras.models.load_model("GRU_model.h5")
        elif model_type == "rnn":
            model = tf.keras.models.load_model("RNN_model.h5")
        elif model_type == "gpt2":
            predicted_sequence = predict_gpt2(input_text, num_words)
            return jsonify({"input_text": input_text, "predicted_text": predicted_sequence})
        else:
            return jsonify({"error": "Invalid model type. Choose 'lstm', 'bi_lstm', 'gru', 'rnn', or 'gpt2'."}), 400

        # Predict the next words
        predicted_sequence = predict_next_words(model, tokenizer, input_text, num_words)
        return jsonify({"input_text": input_text, "predicted_text": predicted_sequence})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
