 # JARVIS: Deep Learning-Based Virtual Assistant 🤖

A personal AI voice assistant built using deep learning and natural language processing techniques. Inspired by Marvel's JARVIS, this assistant can respond to your voice commands, manage schedules, open/close apps, check system status, and engage in casual conversations.

---

## 🔧 Features

- 🎤 **Voice Interaction**: Listens and responds to voice commands using `speech_recognition` and `pyttsx3`.
- 🧠 **Intent Recognition**: Classifies user intents via a trained deep learning model.
- 📚 **Custom Responses**: Replies using predefined patterns and responses in `intents.json`.
- ⌨️ **Keyboard Automation**: Controls system volume and opens/closes native apps.
- 🌐 **Web Browsing**: Opens Google, Facebook, Discord, Instagram via voice commands.
- 📅 **University Schedule**: Speaks daily class schedule based on the current weekday.
- 🔋 **System Monitoring**: Informs battery level and CPU usage.
- 😂 **Jokes & Fun**: Built-in joke responses for entertainment.
- 👤 **Custom Identity**: Introduces itself and its creator (Aditya Dey).

---

## 🧠 Tech Stack

- **Language**: Python 3
- **Deep Learning**: TensorFlow / Keras
- **NLP**: Tokenizer, Label Encoding, Intent Classification
- **Voice Tools**: `speech_recognition`, `pyttsx3`
- **Utilities**: `pyautogui`, `webbrowser`, `psutil`
- **GUI (optional)**: CLI-based interaction

---

## 🗂️ Project Structure

```bash
.
├── main.py                 # Core application logic
├── model_train.py          # Model training script
├── model_test.py           # Text-based testing interface
├── intents.json            # Intent patterns and responses
├── tokenizer.pkl           # Saved tokenizer object
├── label_encoder.pkl       # Saved label encoder
├── chat_model.h5           # Trained Keras model
├── settings.json           # VSCode plugin config (optional)
└── tempCodeRunnerFile.py   # Scratchpad file (ignored)

Install Dependencies
pip install tensorflow pyttsx3 SpeechRecognition numpy pyautogui psutil
Run the Assistant
python main.py

Speak commands like:
"What is the time?"
"Open notepad"
"Tell me a joke"
"Who made you?"
To exit: "exit"

🗣 Sample Intents
greeting: "Hello", "Hi", "Namaste"
datetime: "What time is it?"
jokes: "Tell me a joke"
identity: "Who are you?"
programmer: "Who made you?"

👨‍💻 Creator
Aditya Dey
Passionate about AI, NLP, and building interactive intelligent systems.

## 🧪 Try It Live

- 🌐 [Web Chatbot (Gradio)](https://huggingface.co/spaces/adiv1205/Aditya.JARVIS-Chatbot)  
  A web-based text version of the JARVIS Assistant powered by deep learning.  
  > Built & maintained by Aditya Dey.

Feel free to type your messages and see how JARVIS responds!
