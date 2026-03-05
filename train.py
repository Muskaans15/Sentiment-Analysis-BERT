import torch
from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
dataset = load_dataset("imdb")

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 128
BATCH_SIZE = 16

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

def tokenize(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

dataset = dataset.map(tokenize, batched=True)
dataset = dataset.remove_columns(["text"])
dataset = dataset.rename_column("label", "labels")
dataset.set_format("torch")

train_loader = DataLoader(dataset["train"], batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset["test"], batch_size=BATCH_SIZE)

model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

model.to(device)

optimizer = AdamW(model.parameters(), lr=2e-5)

EPOCHS = 1   # keep 1 for quick test

model.train()
for epoch in range(EPOCHS):
    print(f"Epoch {epoch+1}")
    for batch in tqdm(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

# Save model
model.save_pretrained("bert-imdb-sentiment")
tokenizer.save_pretrained("bert-imdb-sentiment")

print("Model saved successfully!")