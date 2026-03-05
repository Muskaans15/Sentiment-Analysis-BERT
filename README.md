---

title: Sentiment Analysis BERT
emoji: "🤖"
colorFrom: red
colorTo: pink
sdk: gradio
app_file: app.py
pinned: false
license: mit
short_description: Sentiment analysis web app using BERT and Gradio
-------------------------------------------------------------------

# Sentiment Analysis using BERT

This project is a sentiment analysis web application built using a fine-tuned **BERT (Bidirectional Encoder Representations from Transformers)** model. The application analyzes movie reviews and predicts whether the sentiment is **Positive, Negative, Neutral, or Mixed**.

The model was trained on the **IMDB movie review dataset** and integrated into a simple web interface using **Gradio**, allowing users to easily test the model directly from the browser.

## Features

* Analyze the sentiment of a single movie review
* Detect **Positive, Negative, Neutral, and Mixed sentiments**
* Upload a **CSV file containing multiple reviews** for batch analysis
* Download the **results as a new CSV file**

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

## Project Goal

The purpose of this project is to demonstrate how modern **Natural Language Processing (NLP)** models like BERT can be used to build practical applications for analyzing text data and deploying them through a simple web interface.
