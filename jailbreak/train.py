import os, random, numpy as np
from datasets import load_dataset, concatenate_datasets, DatasetDict, Dataset, ClassLabel
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          DataCollatorWithPadding, TrainingArguments, Trainer)
import evaluate
import torch

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# Load all four sub-datasets and concatenate
configs = ["jailbreak_2023_05_07","jailbreak_2023_12_25",
           "regular_2023_05_07","regular_2023_12_25"]
splits = [load_dataset("TrustAIRLab/in-the-wild-jailbreak-prompts", cfg, split="train")
          for cfg in configs]
raw = concatenate_datasets(splits)
df = raw.to_pandas()
df = df.drop_duplicates(subset=["prompt"], keep="first")
raw = Dataset.from_pandas(df, preserve_index=False)

# map jailbreak labels to ClassLabel
raw = raw.rename_column("jailbreak", "label")
raw = raw.remove_columns([c for c in raw.column_names if c not in ["prompt","label"]])
raw = raw.map(lambda x: {"label": x["label"]})
raw = raw.cast_column("label", ClassLabel(names=["not_jailbreak", "jailbreak"]))

# train/val/test split (80/10/10)
tmp = raw.train_test_split(test_size=0.2, seed=SEED, stratify_by_column="label")
test_valid = tmp["test"].train_test_split(test_size=0.5, seed=SEED, stratify_by_column="label")
ds = DatasetDict(train=tmp["train"], valid=test_valid["train"], test=test_valid["test"])

import datasets
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer
)
import evaluate
import torch
from torch import nn

# -------------------------
# TOKENIZATION
# -------------------------
MODEL = "microsoft/deberta-v3-base"
tok = AutoTokenizer.from_pretrained(MODEL)

def tok_fn(batch):
    enc = tok(batch["prompt"], truncation=True, max_length=384)
    enc["labels"] = batch["label"]
    return enc

ds_tok = ds.map(
    tok_fn,
    batched=True,
    remove_columns=ds["train"].column_names
)

# enforce int64 labels
for split in ds_tok.keys():
    ds_tok[split] = ds_tok[split].cast_column("labels", datasets.Value("int64"))

# -------------------------
# CLASS WEIGHTS
# -------------------------
y_train = np.array(ds["train"]["label"])
pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
class_weights = torch.tensor([1.0, float(pos_weight)], dtype=torch.float32)

print("Class weights:", class_weights)

# -------------------------
# METRICS
# -------------------------
metrics = {
    "accuracy": evaluate.load("accuracy"),
    "f1": evaluate.load("f1"),
    "precision": evaluate.load("precision"),
    "recall": evaluate.load("recall"),
    "roc_auc": evaluate.load("roc_auc", "binary")
}

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    preds = (probs >= 0.5).astype(int)
    return {
        "accuracy": metrics["accuracy"].compute(predictions=preds, references=labels)["accuracy"],
        "f1": metrics["f1"].compute(predictions=preds, references=labels, average="binary")["f1"],
        "precision": metrics["precision"].compute(predictions=preds, references=labels, average="binary")["precision"],
        "recall": metrics["recall"].compute(predictions=preds, references=labels, average="binary")["recall"],
        "roc_auc": metrics["roc_auc"].compute(prediction_scores=probs, references=labels)["roc_auc"]
    }

# -------------------------
# MODEL
# -------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL,
    num_labels=2
)

# -------------------------
# CUSTOM TRAINER (weighted CE)
# -------------------------
class WeightedCETrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights  # store the original weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs["labels"]

        # get model device (cuda or cpu)
        device = next(model.parameters()).device

        # move weights to correct device
        weights = self.class_weights.to(device)

        # create the loss fn on the correct device
        loss_fn = nn.CrossEntropyLoss(weight=weights)

        outputs = model(**inputs)
        logits = outputs.logits

        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss

# -------------------------
# COLLATOR
# -------------------------
collate = DataCollatorWithPadding(tokenizer=tok)

# -------------------------
# TRAINING ARGS
# -------------------------
args = TrainingArguments(
    output_dir="./deberta_jailbreak",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    metric_for_best_model="roc_auc",
    load_best_model_at_end=True,
    gradient_accumulation_steps=1,
    fp16=torch.cuda.is_available(),
    report_to="none",
    logging_steps=50
)

# -------------------------
# TRAINER
# -------------------------
trainer = WeightedCETrainer(
    class_weights=class_weights,
    model=model,
    args=args,
    train_dataset=ds_tok["train"],
    eval_dataset=ds_tok["valid"],
    tokenizer=tok,
    data_collator=collate,
    compute_metrics=compute_metrics
)

# -------------------------
# TRAIN & EVAL
# -------------------------
trainer.train()
trainer.evaluate(ds_tok["test"])

# -------------------------
# SAVE BEST MODEL
# -------------------------
trainer.save_model("./deberta_jailbreak/best")
tok.save_pretrained("./deberta_jailbreak/best")