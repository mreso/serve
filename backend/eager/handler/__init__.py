import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class Handler(object):
    def __init__(self):
        self._tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc", use_fast=False)

        self._model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased-finetuned-mrpc")


    def execute(self, sequence_0, sequence_1):
        token_ids_mask = self._tokenizer.encode_plus(sequence_0, sequence_1, return_tensors="pt")


        classification_logits = self._model(**token_ids_mask)[0]

        paraphrase_results = torch.softmax(classification_logits, dim=1).tolist()[0]

        return f"{round(paraphrase_results[1] * 100)}% paraphrase"
