#!/usr/bin/env python
# coding: utf-8

import os
import time
from unittest.mock import patch

import torch
from torch import package
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sequence_0 = "The company HuggingFace is based in New York City"
sequence_1 = "Apples are especially bad for your health"
sequence_2 = "HuggingFace's headquarters are situated in Manhattan"

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc", use_fast=False)

def run_model(model_, s0, s1):
    token_ids_mask = tokenizer.encode_plus(s0, s1, return_tensors="pt")

    token_ids_mask = {k:v.to(model.device) for k,v in token_ids_mask.items()}

    st = time.time()
    classification_logits = model_(**token_ids_mask)
    print(f"Execution time: {1000*(time.time() - st):.2f} ms")

    paraphrase_results = torch.softmax(classification_logits[0], dim=1).cpu().tolist()[0]

    return f"{round(paraphrase_results[1] * 100)}% paraphrase"


model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased-finetuned-mrpc", torchscript=True)
model.eval()

if torch.cuda.is_available():
  torch.cuda.init()
  model = model.to('cuda')

print('Original Model')
print(run_model(model, sequence_0, sequence_1))
print(run_model(model, sequence_0, sequence_2))

base_path = 'models/bert_model_only'

## Export with TorchPackage
package_path = f'{base_path}.pt'

with package.PackageExporter(package_path) as bert_package_exp:
    bert_package_exp.intern([
      'model.**',
      ])
    bert_package_exp.extern([
      'torch.**',
      'sys',
      'io',
      '__future__.**',
      '_queue',
      'transformers.**',
      'packaging.**',
      'importlib_metadata.**',
      'tokenizers.**',
      'torchaudio.**',
      'huggingface_hub',
      'PIL.**',
      'yaml.**',
      'numpy.**',
      'urllib3.**',
      'requests.**',
      'pkg_resources.**',
      'regex.**',
      'six.**',
      'sacremoses.**',
      'absl.**',
      'idna.**',
      'tqdm.**',
      'filelock.**',
      'google.**',
      'IPython.display.**',
      'certifi.**',
      'charset_normalizer.**',
    ])

    bert_package_exp.save_pickle("model", "model.pkl", model)

## Export with TorchScript
max_sequence_length = 64

dummy_input = [
        torch.zeros([1, max_sequence_length], dtype=torch.long, device=model.device),
        torch.zeros([1, max_sequence_length], dtype=torch.long, device=model.device),
        torch.ones([1, max_sequence_length], dtype=torch.long, device=model.device),
        ]

traced_model = torch.jit.trace(model, dummy_input)
traced_path = f'{base_path}_traced.pt'
torch.jit.save(traced_model, traced_path)


from torch import package
# Above code saves packaged resnet to disk.  To load, and execute:
bert_package_imp = package.PackageImporter(package_path)

pickled_model = bert_package_imp.load_pickle("model", "model.pkl")

print('TorchPackaged Model')
print(run_model(pickled_model, sequence_0, sequence_1))
print(run_model(pickled_model, sequence_0, sequence_2))

loaded_traced_model = torch.jit.load(traced_path)


print('Traced Model')
print(run_model(loaded_traced_model, sequence_0, sequence_1))
print(run_model(loaded_traced_model, sequence_0, sequence_2))
