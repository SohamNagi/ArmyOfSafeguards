## Load huggingface model and run inference
import torch
from transformers import (
    AutoTokenizer, AutoConfig,
    AutoModelForSequenceClassification, PreTrainedModel
)
from huggingface_hub import login, hf_hub_download
from safetensors.torch import load_file
from torch import nn

repo_id = "tommypang04/finetuned-model-jailbrak"
tok = AutoTokenizer.from_pretrained(repo_id)
model = AutoModelForSequenceClassification.from_pretrained(repo_id)
model.eval()

def predict(text: str):
    enc = tok(text, return_tensors="pt", truncation=True, max_length=384)

    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).squeeze()

        pred_id = torch.argmax(probs).item()
        confidence = probs[pred_id].item()
        
        return {
            "label": bool(pred_id), # idx 0 is False, 1 is True, labels are [not_jailbreak, jailbreak]
            "confidence": confidence,
        }
