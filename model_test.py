import json
import pickle
import numpy as np
import random
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load intents
with open("intents.json") as file:
    data = json.load(file)

# Load model and tools
model = load_model("chat_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Chat loop
while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "exit"]:
        print("Bot: Goodbye!")
        break

    sequence = tokenizer.texts_to_sequences([user_input])
    padded = pad_sequences(sequence, truncating='post', maxlen=20)

    prediction = model.predict(padded)
    tag = label_encoder.inverse_transform([np.argmax(prediction)])

    for intent in data['intents']:
        if intent['tag'] == tag:
            print("Bot:", random.choice(intent['responses']))
