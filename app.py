import gradio as gr
import json
import pickle
import numpy as np
import random
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load intents and model
with open("intents.json") as file:
    data = json.load(file)

model = load_model("chat_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Chatbot response logic
def chat(user_input):
    if user_input.lower() in ["exit", "quit"]:
        return "Goodbye!"

    sequence = tokenizer.texts_to_sequences([user_input])
    padded = pad_sequences(sequence, truncating='post', maxlen=20)

    prediction = model.predict(padded)
    tag = label_encoder.inverse_transform([np.argmax(prediction)])[0]

    for intent in data['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

    return "Sorry, I didn't understand that."

# Gradio UI
iface = gr.Interface(
    fn=chat,
    inputs="text",
    outputs="text",
    title="JARVIS Chatbot",
    description="A Deep Learning-based intelligent assistant by Aditya Dey. Just type and chat!",
)

iface.launch()
