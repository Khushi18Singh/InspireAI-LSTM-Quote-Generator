# ✨ Khushi's InspireAI — LSTM Quote Generator Platform

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Model](https://img.shields.io/badge/Model-LSTM%20(Deep%20Learning)-purple)
![Framework](https://img.shields.io/badge/Framework-TensorFlow%20%7C%20Keras-orange)

A Deep Learning powered NLP web app that generates original, inspirational quotes word-by-word using a custom-trained **LSTM (Long Short-Term Memory) Neural Network**, packaged into a full multi-tool creative studio built with **Streamlit**.

🔗 **Live Demo:** _(add your Streamlit deployment link here once deployed)_

---

## 📑 Table of Contents

- [About the Project](#-about-the-project)
- [Project Workflow](#-project-workflow)
- [Dataset](#-dataset)
- [Live Features](#-live-features)
- [Deep Learning Model](#-deep-learning-model)
- [Application Preview](#️-application-preview)
- [How to Use](#-how-to-use)
- [Tech Stack](#️-tech-stack)
- [Setup](#-setup)
- [Project Structure](#-project-structure)
- [Future Improvements](#-future-improvements)
- [Developer](#-developer)

---

## 📖 About the Project

- Generates brand-new, original quotes from a seed phrase using a custom-trained LSTM neural network (not a lookup or template system).
- Goes beyond text generation — bundles the model into a complete creative suite: poster design studio, AI-vs-Human quiz game, text-to-speech narration, and an e-book compiler.
- Includes a dedicated **Deep Learning Lab** tab that visually breaks down the model's architecture and tokenizer pipeline for anyone curious about how it works under the hood.
- Packaged as a full interactive Streamlit application, not just a training notebook.

---

## 🔄 Project Workflow

1. **Input** — a seed phrase typed by the user (e.g. *"life is a"*, *"courage means"*)
2. **Tokenization** — seed text converted into integer tokens using a fitted Keras `Tokenizer`
3. **Padding** — token sequence padded to the model's fixed input length (`max_len`)
4. **Generation** — the trained LSTM model predicts the next word repeatedly, word-by-word, using configurable sampling strategies (temperature / top-k / top-p) to control creativity
5. **Post-processing** — generated words assembled into a complete, capitalized quote
6. **Output** — final quote with sentiment/mood analysis, a designed poster card, audio narration, and social-media-ready captions

---

## 📊 Dataset

- **Name:** Quotes Dataset (`qoute_dataset.csv`) — not included in this repository
- **Content:** Thousands of curated inspirational/motivational quotes used to train the tokenizer and the LSTM language model
- **Usage:** Used offline during training to build the vocabulary (~8,978 unique words) and fit the sequence model; the trained artifacts (`lstm_model.h5`, `tokenizer.pkl`, `max_len.pkl`) are what the app actually loads at runtime

> 📌 The raw dataset file is kept out of this repository. Only the trained model, tokenizer, and max-length artifacts are used by the deployed app.

---

## ⚡ Live Features

- ✨ **AI Studio** — generate original quotes from a seed phrase with adjustable creativity (temperature), sampling mode (top-k / top-p), and word count
- 😊 **Sentiment & Mood Analysis** — every generated quote is analyzed for mood/sentiment
- 🎨 **Poster Studio** — turn any generated quote into a beautifully designed, downloadable quote poster (PNG)
- 🔊 **Text-to-Speech Narration** — listen to any generated quote read aloud
- 🎮 **AI vs Human Quiz** — a guessing game to test whether a quote was written by AI or a human author
- 🏛️ **Poster Gallery** — browse a curated exhibition of pre-designed quote posters
- 🎓 **Deep Learning Lab** — interactive tokenizer simulator + a visual, step-by-step breakdown of the LSTM architecture (embedding → LSTM layers → dropout → softmax output)
- 📖 **Quote Book Creator** — compile saved favorite quotes into a downloadable, stylized HTML e-book
- 📊 **Analytics & Dataset Explorer** — word cloud of most frequent words, top-quoted authors, and a searchable dataset browser
- 📲 **One-Click Social Formatting** — auto-generated Instagram, Twitter/X, and LinkedIn-ready captions for any quote
- 🌗 **Light / Dark / System theme** support with a fully custom glassmorphism UI

---

## 🧠 Deep Learning Model

| Component | Detail |
|---|---|
| Architecture | Embedding → LSTM (128 units) → Dropout (0.2) → Dense (Softmax) |
| Vocabulary Size | ~8,978 unique words |
| Framework | TensorFlow / Keras |
| Task | Next-word prediction (sequence generation) |
| Sampling Strategies | Temperature scaling, Top-K sampling, Top-P (nucleus) sampling |

**Why LSTM?** Recurrent memory cells allow the model to retain context across a sequence of words, making it well suited to generating grammatically coherent, contextually relevant short-form text like quotes.

---

## 🖼️ Application Preview

![App Screenshot](screenshot.png)

> Add a screenshot of your running app here — save it as `screenshot.png` in the project root, and it will automatically render above once pushed to GitHub.

---

## 🎮 How to Use

1. Run the app → opens in your browser
2. **AI Studio tab** — type or pick a seed phrase → adjust creativity/sampling settings → click Generate Quote
3. **Poster Studio tab** — design and download a custom quote poster
4. **AI vs Human Quiz tab** — play the guessing game and track your score/streak
5. **Poster Gallery tab** — browse curated quote poster designs
6. **DL Model Lab tab** — explore the tokenizer simulator and LSTM architecture breakdown
7. **Quote Book Creator tab** — export your saved favorite quotes as a downloadable HTML e-book
8. **Analytics & Dataset tab** — explore word frequency, top authors, and search the dataset

---

## 🛠️ Tech Stack

- **Streamlit** — web app framework
- **TensorFlow / Keras** — LSTM model training & inference
- **NumPy & Pandas** — data handling
- **gTTS** — text-to-speech audio generation
- **TextBlob** — sentiment/mood analysis
- **Matplotlib & WordCloud** — analytics visualizations
- **Pillow (PIL)** — poster/image generation

---

## 🚀 Setup

```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
streamlit run app.py
```

> Note: the training dataset (`qoute_dataset.csv`) is not included in this repo. The app runs fully on the pre-trained model artifacts (`lstm_model.h5`, `tokenizer.pkl`, `max_len.pkl`); the Analytics tab will simply be unavailable without the dataset file.

---

## 📂 Project Structure

```
├── app.py               # Main Streamlit application
├── lstm_model.h5          # Trained LSTM language model
├── tokenizer.pkl          # Fitted Keras tokenizer
├── max_len.pkl            # Model's fixed input sequence length
├── requirements.txt       # Python dependencies
├── .gitignore
└── README.md
```

---

## 🔮 Future Improvements

- Upgrade to a Transformer-based architecture for richer, more diverse generations
- Fine-tune on author-specific styles for personalized quote generation
- Add multi-language quote generation support
- Persistent, database-backed favorites and history (not session-only)
- Deploy on Streamlit Cloud / Docker
- Add model explainability for token-level generation decisions

---

## 👩‍💻 Developer

**Khushi Singh** — Data Scientist · Machine Learning Engineer · Data Analyst