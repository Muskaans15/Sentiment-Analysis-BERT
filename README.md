---

title: Sentiment Analysis BERT
emoji: "🤖"
colorFrom: red
colorTo: pink
sdk: gradio
app_file: app.py
pinned: false


# Sentiment Analysis using BERT

This project is a sentiment analysis web application built using a fine-tuned **BERT (Bidirectional Encoder Representations from Transformers)** model.

The application analyzes movie reviews and predicts whether the sentiment is **Positive, Negative, Neutral, or Mixed**.

The model was trained on the **IMDB movie review dataset** and integrated into a simple web interface using **Gradio**, allowing users to test the model directly from the browser.

## Features

* Analyze the sentiment of a single movie review
* Detect **Positive, Negative, Neutral, and Mixed sentiments**
* Upload a **CSV file containing multiple reviews**
* Download the **results as a CSV file**

## Technologies Used

* Python
* PyTorch
* Hugging Face Transformers
* Gradio
* Pandas

## Running the Project Locally

Install dependencies:

pip install -r requirements.txt

Run the app:

python app.py

Open in browser:

http://127.0.0.1:7860

---