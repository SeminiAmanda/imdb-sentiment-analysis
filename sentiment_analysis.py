# ============================================================
# STEP 1 — Install & Import
# ============================================================
# Run once in terminal:
#   pip install -r requirements.txt

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# HuggingFace + PyTorch
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ── Device check ──
# ── Device check (Updated for Mac GPU) ──
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")
# Expected output on Mac: "Using device: mps"


# ============================================================
# STEP 2 — Text Cleaning Utility
# ============================================================
# Must be defined early — used in EDA, tokenization, inference.

def clean_text(text):
    text = re.sub(r'<.*?>', ' ', text)       # remove HTML tags e.g. <br />
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # remove punctuation & numbers
    text = re.sub(r'\s+', ' ', text)          # collapse extra whitespace
    return text.lower().strip()


# ============================================================
# STEP 3 — Load Dataset from Local CSV
# ============================================================
# Download from Kaggle:
# kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
#
# Put "IMDB Dataset.csv" in the same folder as this script.
# The CSV has two columns:  review | sentiment
# sentiment values: "positive" or "negative"

print("Loading IMDB dataset from local CSV...")

CSV_PATH = "/Users/amandasewwandi/Library/CloudStorage/OneDrive-sci.sjp.ac.lk/NLP_Project/rawData/IMDB Dataset.csv"   # ← change this if your filename differs

df = pd.read_csv(CSV_PATH)

print(f"Shape      : {df.shape}")           # (50000, 2)
print(f"Columns    : {list(df.columns)}")   # ['review', 'sentiment']
print(f"\nLabel counts:\n{df['sentiment'].value_counts()}")

# ── Convert string labels → 0 / 1 ──
df["label"] = df["sentiment"].map({"negative": 0, "positive": 1})
df = df[["review", "label"]].rename(columns={"review": "text"})

# ── Train / test split  80 / 20 ──
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label"]
)
train_df = train_df.reset_index(drop=True)
test_df  = test_df.reset_index(drop=True)

print(f"\nTrain size : {len(train_df)}")    # 40,000
print(f"Test size  : {len(test_df)}")       # 10,000

# ── Wrap in HuggingFace DatasetDict ──
raw_dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df),
    "test":  Dataset.from_pandas(test_df),
})
print("\nDataset ready:")
print(raw_dataset)

# ── Quick peek ──
sample = raw_dataset["train"][0]
print("\nSample review (first 300 chars):")
print(sample["text"][:300])
print(f"Label: {sample['label']}  (0=Negative, 1=Positive)")


# ============================================================
# STEP 4 — Exploratory Data Analysis (EDA)
# ============================================================

# ── 4a. Label distribution + review length ──
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

label_names = ["Negative", "Positive"]
counts = train_df["label"].value_counts().sort_index()
axes[0].bar(label_names, counts.values, color=["#e74c3c", "#2ecc71"], width=0.4)
axes[0].set_title("Label Distribution (Train)", fontsize=13)
axes[0].set_ylabel("Count")
for i, v in enumerate(counts.values):
    axes[0].text(i, v + 100, str(v), ha="center", fontweight="bold")

train_df["length"] = train_df["text"].apply(lambda x: len(x.split()))
axes[1].hist(train_df["length"], bins=60, color="#7c6de0", edgecolor="white", alpha=0.85)
axes[1].set_title("Review Length (words)", fontsize=13)
axes[1].set_xlabel("Word count")
axes[1].set_ylabel("Frequency")
axes[1].axvline(
    train_df["length"].median(), color="orange", linestyle="--",
    label=f'Median: {train_df["length"].median():.0f}'
)
axes[1].legend()

plt.tight_layout()
plt.savefig("eda_overview.png", dpi=150, bbox_inches="tight")
#plt.show()
print(f"Average review length : {train_df['length'].mean():.0f} words")
print(f"Max review length     : {train_df['length'].max()} words")

# ── 4b. Word clouds (with HTML cleaning) ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, lbl, title, color in zip(
    axes,
    [0, 1],
    ["Negative Reviews", "Positive Reviews"],
    ["Reds", "Greens"],
):
    texts = " ".join(
        train_df[train_df["label"] == lbl]["text"]
        .sample(500, random_state=42)
        .apply(clean_text)          # ← removes <br> and punctuation
    )
    wc = WordCloud(
        width=600, height=300,
        colormap=color,
        background_color="white",
        max_words=150,
    ).generate(texts)
    ax.imshow(wc, interpolation="bilinear")
    ax.set_title(title, fontsize=13)
    ax.axis("off")

plt.tight_layout()
plt.savefig("wordclouds.png", dpi=150, bbox_inches="tight")
#plt.show()


# ============================================================
# STEP 5 — Tokenization with DistilBERT
# ============================================================
# DistilBERT = smaller, faster BERT (66M params vs 110M).
# Runs fine on free Google Colab T4 GPU.
# To use full BERT: change MODEL_NAME = "bert-base-uncased"

MODEL_NAME  = "distilbert-base-uncased"
MAX_LENGTH  = 256     # covers ~85% of reviews without truncation
BATCH_SIZE  = 16
EPOCHS      = 3
LR          = 2e-5

print(f"\nLoading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# ── Inspect one tokenisation ──
example = "This movie was absolutely fantastic — a real masterpiece!"
tokens  = tokenizer(example, return_tensors="pt")
print("Tokenized example:")
print("Decoded:", tokenizer.convert_ids_to_tokens(tokens["input_ids"][0]))

# ── Tokenize full dataset (clean text first) ──
def tokenize_fn(batch):
    cleaned = [clean_text(t) for t in batch["text"]]   # remove HTML/punctuation
    return tokenizer(
        cleaned,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
    )

print("\nTokenizing dataset (takes ~1–2 min)...")
tokenized = raw_dataset.map(tokenize_fn, batched=True, batch_size=1000)
tokenized = tokenized.rename_column("label", "labels")
tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# ── Subset for faster training ──
# On free Colab GPU:  set TRAIN_SIZE=25000, EVAL_SIZE=5000
# On CPU only:        keep TRAIN_SIZE=5000 (takes ~2 hrs even then)
TRAIN_SIZE = 5000    # ← increase to 25000 on GPU for best accuracy
EVAL_SIZE  = 1000    # ← increase to 5000  on GPU

train_subset = tokenized["train"].shuffle(seed=42).select(range(TRAIN_SIZE))
eval_subset  = tokenized["test"].shuffle(seed=42).select(range(EVAL_SIZE))

print(f"\nTraining on   : {TRAIN_SIZE} samples")
print(f"Evaluating on : {EVAL_SIZE} samples")


# ============================================================
# STEP 6 — Load Pre-trained Model
# ============================================================
# num_labels=2  → binary (positive / negative)
# The classification head is randomly initialised.
# All DistilBERT weights are pre-trained on Wikipedia + BookCorpus.

print(f"\nLoading model: {MODEL_NAME}")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
)
model.to(device)

total_params     = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters     : {total_params:,}")
print(f"Trainable parameters : {trainable_params:,}")


# ============================================================
# STEP 7 — Metrics
# ============================================================

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1":       f1_score(labels, preds, average="weighted"),
    }


# ============================================================
# STEP 8 — Training Configuration
# ============================================================

training_args = TrainingArguments(
    output_dir                  = "./results",
    num_train_epochs            = EPOCHS,
    per_device_train_batch_size = BATCH_SIZE,
    per_device_eval_batch_size  = BATCH_SIZE,
    learning_rate               = LR,
    weight_decay                = 0.01,
    warmup_ratio                = 0.1,           # 10% of steps for LR warmup
    evaluation_strategy         = "epoch",       # evaluate at end of each epoch
    save_strategy               = "epoch",
    load_best_model_at_end      = True,
    metric_for_best_model       = "f1",
    logging_dir                 = "./logs",
    logging_steps               = 50,
    fp16                        = torch.cuda.is_available(),  # mixed precision on GPU
    report_to                   = "none",        # disables wandb / tensorboard
    seed                        = 42,
)

trainer = Trainer(
    model           = model,
    args            = training_args,
    train_dataset   = train_subset,
    eval_dataset    = eval_subset,
    compute_metrics = compute_metrics,
    callbacks       = [EarlyStoppingCallback(early_stopping_patience=2)],
)


# ============================================================
# STEP 9 — Fine-tune!
# ============================================================
# Colab T4 GPU  + 5K samples  →  ~4 minutes
# Colab T4 GPU  + 25K samples →  ~25 minutes
# CPU only      + 5K samples  →  ~2 hours (not recommended)

print("\nStarting fine-tuning...")
train_result = trainer.train()

print("\nTraining complete!")
print(f"Training loss : {train_result.training_loss:.4f}")
print(f"Runtime       : {train_result.metrics['train_runtime']:.1f}s")


# ============================================================
# STEP 10 — Evaluate on Test Set
# ============================================================

print("\nEvaluating on test set...")
eval_results = trainer.evaluate()

print(f"\n{'='*42}")
print(f"  Accuracy : {eval_results['eval_accuracy']:.4f}")
print(f"  F1 Score : {eval_results['eval_f1']:.4f}")
print(f"  Eval Loss: {eval_results['eval_loss']:.4f}")
print(f"{'='*42}")

# ── Full classification report ──
predictions = trainer.predict(eval_subset)
preds  = np.argmax(predictions.predictions, axis=-1)
labels = predictions.label_ids

print("\nClassification Report:")
print(classification_report(labels, preds, target_names=["Negative", "Positive"]))


# ============================================================
# STEP 11 — Visualise Results
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# ── 11a. Confusion matrix ──
cm = confusion_matrix(labels, preds)
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Purples",
    xticklabels=["Negative", "Positive"],
    yticklabels=["Negative", "Positive"],
    ax=axes[0], linewidths=0.5,
)
axes[0].set_title("Confusion Matrix", fontsize=13)
axes[0].set_ylabel("Actual")
axes[0].set_xlabel("Predicted")

# ── 11b. Training & validation loss curve ──
log_history  = trainer.state.log_history
train_losses = [x["loss"]      for x in log_history if "loss"      in x and "eval_loss" not in x]
eval_losses  = [x["eval_loss"] for x in log_history if "eval_loss" in x]

if train_losses:
    axes[1].plot(train_losses, label="Train loss", color="#7c6de0")
if eval_losses:
    step_positions = [
        len(train_losses) // len(eval_losses) * (i + 1)
        for i in range(len(eval_losses))
    ]
    axes[1].plot(step_positions, eval_losses, label="Val loss",
                 color="#e74c3c", marker="o")
axes[1].set_title("Training & Validation Loss", fontsize=13)
axes[1].set_xlabel("Step")
axes[1].set_ylabel("Loss")
axes[1].legend()

plt.tight_layout()
plt.savefig("results.png", dpi=150, bbox_inches="tight")
plt.show()


# ============================================================
# STEP 12 — Baseline Comparison  (TF-IDF + Logistic Regression)
# ============================================================
# Shows exactly how much BERT improves over classical ML.

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

print("\nTraining TF-IDF baseline for comparison...")

train_texts  = [clean_text(raw_dataset["train"][i]["text"])  for i in range(TRAIN_SIZE)]
train_labels = [raw_dataset["train"][i]["label"]             for i in range(TRAIN_SIZE)]
test_texts   = [clean_text(raw_dataset["test"][i]["text"])   for i in range(EVAL_SIZE)]
test_labels  = [raw_dataset["test"][i]["label"]              for i in range(EVAL_SIZE)]

tfidf   = TfidfVectorizer(max_features=30000, ngram_range=(1, 2), sublinear_tf=True)
X_train = tfidf.fit_transform(train_texts)
X_test  = tfidf.transform(test_texts)

lr      = LogisticRegression(max_iter=500, C=1.0)
lr.fit(X_train, train_labels)
lr_preds = lr.predict(X_test)
lr_acc   = accuracy_score(test_labels, lr_preds)
lr_f1    = f1_score(test_labels, lr_preds, average="weighted")

print(f"\n{'Model':<30} {'Accuracy':>10} {'F1 Score':>10}")
print("-" * 54)
print(f"{'TF-IDF + Logistic Reg':<30} {lr_acc:>10.4f} {lr_f1:>10.4f}")
print(f"{'DistilBERT (fine-tuned)':<30} {eval_results['eval_accuracy']:>10.4f} {eval_results['eval_f1']:>10.4f}")
print("-" * 54)
improvement = (eval_results["eval_accuracy"] - lr_acc) * 100
print(f"BERT improvement: +{improvement:.2f}%")


# ============================================================
# STEP 13 — Save the Model
# ============================================================

MODEL_SAVE_PATH = "./sentiment_model"

print(f"\nSaving model to {MODEL_SAVE_PATH}...")
trainer.save_model(MODEL_SAVE_PATH)
tokenizer.save_pretrained(MODEL_SAVE_PATH)
print("Saved: config.json, model.safetensors, tokenizer files")


# ============================================================
# STEP 14 — Inference on Custom Text
# ============================================================

def predict_sentiment(text: str, model_path: str = MODEL_SAVE_PATH) -> dict:
    """
    Load saved model and predict sentiment of any text string.

    Returns dict:
        label          → "POSITIVE" or "NEGATIVE"
        confidence     → float 0–1
        probabilities  → {"POSITIVE": float, "NEGATIVE": float}
    """
    tok = AutoTokenizer.from_pretrained(model_path)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_path)
    mdl.eval()
    mdl.to(device)

    inputs = tok(
        clean_text(text),
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
    ).to(device)

    with torch.no_grad():
        outputs = mdl(**inputs)

    probs  = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]
    pred   = int(np.argmax(probs))
    labels_map = ["NEGATIVE", "POSITIVE"]

    return {
        "label":         labels_map[pred],
        "confidence":    float(probs[pred]),
        "probabilities": {
            "NEGATIVE": float(probs[0]),
            "POSITIVE": float(probs[1]),
        },
    }


# ── Test on sample reviews ──
test_reviews = [
    "This movie was absolutely brilliant! One of the best I've ever seen.",
    "Complete waste of time. Terrible acting and a nonsensical plot.",
    "It was okay, not great but not terrible either.",
    "The cinematography was stunning but the story fell flat.",
    "An emotional rollercoaster — I laughed, cried, and cheered.",
]

print("\n" + "=" * 58)
print("  INFERENCE RESULTS")
print("=" * 58)
for review in test_reviews:
    result = predict_sentiment(review)
    emoji  = "✅" if result["label"] == "POSITIVE" else "❌"
    short  = review[:55] + "..." if len(review) > 55 else review
    print(f"\n{emoji}  {result['label']}  ({result['confidence']*100:.1f}% confident)")
    print(f"    \"{short}\"")
    print(f"    P(pos)={result['probabilities']['POSITIVE']:.4f}  "
          f"P(neg)={result['probabilities']['NEGATIVE']:.4f}")

print("\n" + "=" * 58)
print("  ALL STEPS COMPLETE!")
print("  Model saved to  → ./sentiment_model/")
print("  Charts saved to → eda_overview.png, wordclouds.png, results.png")
print("  Next: run  →  streamlit run app.py")
print("=" * 58)