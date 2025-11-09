# Factuality Critic

The factuality safeguard critic flags model outputs that contradict verified facts or propagate misinformation. It is powered by the fine-tuned [ajith-bondili/deberta-v3-factuality-small](https://huggingface.co/ajith-bondili/deberta-v3-factuality-small) DeBERTa-v3 sequence classifier.

## Usage

Install the dependencies listed in `../requirements.txt`, then run the critic as a module:

```bash
python -m factuality.safeguard_factuality "The earth orbits the sun once every 365 days."
```

The command prints the predicted label (`FACTUAL` or `MISINFORMATION`) along with the model confidence so you can integrate the safeguard into your evaluation pipeline.
