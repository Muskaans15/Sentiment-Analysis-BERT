# Sentiment Analysis using BERT

This project is a **Sentiment Analysis Web Application** built using a fine-tuned **BERT (Bidirectional Encoder Representations from Transformers)** model.

The application analyzes movie reviews and predicts whether the sentiment is **Positive, Negative, Neutral, or Mixed**.

The model is trained on the **IMDB movie review dataset** and deployed with a **Gradio web interface**, allowing users to interact with the model directly through a browser.

---

# Live Demo

You can try the live application here:

**Hugging Face App:**
https://huggingface.co/spaces/musk15/sentiment-analysis-bert

---

# Features

* Analyze the sentiment of a single movie review
* Detect **Positive, Negative, Neutral, and Mixed sentiments**
* Upload a **CSV file containing multiple reviews** for batch analysis
* Download the **results as a new CSV file**

---

# Technologies Used

* Python
* PyTorch
* Hugging Face Transformers
* Gradio
* Pandas

---

# Project Structure

```
sentiment-analysis-app
│
├── app.py                # Gradio web application
├── train.py              # Model training script
├── requirements.txt      # Project dependencies
├── README.md             # Project documentation
└── .gitignore            # Ignored files
```

---

# Running the Project Locally

### 1 Install dependencies

```
pip install -r requirements.txt
```

### 2 Run the application

```
python app.py
```

### 3 Open the browser

```
http://127.0.0.1:7860
```

---

# Project Goal

The purpose of this project is to demonstrate how **modern NLP models like BERT** can be used to build practical applications for analyzing text data.

It also shows how machine learning models can be deployed through a **simple web interface**, making AI tools accessible to non-technical users.

---

# Author

Muskaan Singh

