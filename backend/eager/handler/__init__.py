import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class Handler(object):
    def __init__(self, torchscript=False):
        self._tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc", use_fast=False)

        self._model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased-finetuned-mrpc", torchscript=True)

        if torch.cuda.is_available():
            self._model = self._model.to('cuda')


    def execute(self, sequence_0, sequence_1):
        token_ids_mask = self._tokenizer.encode_plus(sequence_0, sequence_1, return_tensors="pt")

        token_ids_mask = {k: v.to(self._model.device) for k,v in token_ids_mask.items()}

        classification_logits = self._model(**token_ids_mask)[0]

        paraphrase_results = torch.softmax(classification_logits, dim=1).cpu().tolist()[0]

        return f"{round(paraphrase_results[1] * 100)}% paraphrase"

    def __call__(self, *args, **kwargs):
        assert len(args) == 1, "Expecting one input argument"

        request = json.loads(args[0])

        assert 'sequence_0' in request and 'sequence_1' in request, "Incorrect JSON content"

        return self.execute(request['sequence_0'], request['sequence_1'])


class InferenceModel(object):
    def __init__(self):
        self._model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased-finetuned-mrpc", torchscript=True)
        self._model.eval()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        print("forward")
        with torch.inference_mode():
           return self._model(*args, **kwargs)

    def to(self, device):
        self._model = self._model.to(device)
        return self
    
    def eval(self):
        self._model.eval()

    def __getattr__(self, key):
        if key in "device":
            return self._model.device
        
        if key in self.__dict__:
            return self.__dict__[key]
        else:
            raise AttributeError


class NoGradModel(object):
    def __init__(self):
        self._model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased-finetuned-mrpc", torchscript=True)
        self._model.eval()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        print("forward")
        with torch.no_grad():
           return self._model(*args, **kwargs)

    def to(self, device):
        self._model = self._model.to(device)
        return self
    
    def eval(self):
        self._model.eval()

    def __getattr__(self, key):
        if key in "device":
            return self._model.device
        
        if key in self.__dict__:
            return self.__dict__[key]
        else:
            raise AttributeError