import torch
import gradio as gr
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

model.to(device)
model.eval()


# -------------------------------
# Sentiment Logic
# -------------------------------

def get_sentiment(text):

    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )

    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]

    neg = probs[0].item()
    pos = probs[1].item()

    diff = abs(pos - neg)
    text_lower = text.lower()

    if "but" in text_lower or "however" in text_lower:
        sentiment = "Mixed Feelings"

    elif diff < 0.10:
        sentiment = "Neutral"

    elif pos > neg:
        sentiment = "Positive"

    else:
        sentiment = "Negative"

    return sentiment


# -------------------------------
# Single Text Prediction
# -------------------------------

def predict_text(text):

    sentiment = get_sentiment(text)

    return f"Sentiment: {sentiment}"


# -------------------------------
# CSV Batch Prediction
# -------------------------------

def analyze_csv(file):

    df = pd.read_csv(file.name)

    if "review" not in df.columns:
        return "CSV must contain column named 'review'"

    sentiments = []

    for text in df["review"]:
        sentiment = get_sentiment(str(text))
        sentiments.append(sentiment)

    df["Sentiment"] = sentiments

    output_file = "sentiment_results.csv"
    df.to_csv(output_file, index=False)

    return output_file


# -------------------------------
# UI
# -------------------------------

with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="violet",
        neutral_hue="slate"
    )
) as demo:

    gr.Markdown(
        """
        # 🎬 AI Sentiment Analyzer  
        ### Analyze movie reviews using a **BERT NLP model**
        """
    )

    gr.Markdown("---")

    with gr.Tabs():

        # ---------------------------
        # TEXT TAB
        # ---------------------------

        with gr.Tab("📝 Text Analysis"):

            with gr.Row():

                with gr.Column(scale=2):

                    review_input = gr.Textbox(
                        label="Enter Review",
                        lines=4,
                        placeholder="Example: The movie was amazing but the ending was disappointing."
                    )

                    analyze_btn = gr.Button(
                        "🔍 Analyze Sentiment",
                        variant="primary"
                    )

                with gr.Column(scale=1):

                    output_text = gr.Textbox(
                        label="Result",
                        lines=2
                    )

            analyze_btn.click(
                predict_text,
                inputs=review_input,
                outputs=output_text
            )


        # ---------------------------
        # CSV TAB
        # ---------------------------

        with gr.Tab("📂 CSV Analysis"):

            gr.Markdown(
                "Upload a CSV file containing a **review column** to analyze sentiments in bulk."
            )

            with gr.Row():

                file_input = gr.File(label="Upload CSV File")

                output_file = gr.File(label="Download Results")

            file_input.change(
                analyze_csv,
                inputs=file_input,
                outputs=output_file
            )

    gr.Markdown(
        """
        ---
        ⚡ Built using **BERT + PyTorch + Gradio**

        """
    )

demo.launch()