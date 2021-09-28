import json
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

    def __call__(self, *args, **kwargs):
        assert len(args) == 1, "Expecting one input argument"

        request = json.loads(args[0])

        assert 'sequence_0' in request and 'sequence_1' in request, "Incorrect JSON content"

        return self.execute(request['sequence_0'], request['sequence_1'])
