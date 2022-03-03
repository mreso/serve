#!/usr/bin/env python
# coding: utf-8

import time

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


tokenizer = AutoTokenizer.from_pretrained("bert-base-cased-finetuned-mrpc", use_fast=True)  

def run_model(model_, s0, s1):
    start_time = time.time()
    token_ids_mask = {}
    for i in range(8):
      tim = tokenizer.encode_plus(s0, s1, return_tensors="pt", max_length=64, padding='max_length')
      for k,v in tim.items():
        if k not in token_ids_mask:
          token_ids_mask[k] = [v]
        else:
          token_ids_mask[k].append(v)

    token_ids_mask = {k:torch.stack(v).squeeze().to(model.device) for k,v in token_ids_mask.items()}

    stop_time = time.time()
    tokenize_time = round((stop_time - start_time) * 1000, 2)
    # tokenize_time = stop_time - start_time

    start_time = time.time()
    classification_logits = model_(**token_ids_mask)
    stop_time = time.time()
    # model_time = stop_time - start_time
    model_time = round((stop_time - start_time) * 1000, 2)

    paraphrase_results = torch.softmax(classification_logits[0], dim=1).cpu().tolist()[0]

    # return f"{round(paraphrase_results[1] * 1000)}% paraphrase"
    return tokenize_time, model_time, f"{round(paraphrase_results[1] * 1000)}% paraphrase"

def benchmark_model(model_, N):
    s0 = "The company HuggingFace is based in New York City"
    s1 = "Apples are especially bad for your health"
    token_ids_mask = tokenizer.encode_plus(s0, s1, return_tensors="pt")

    # Warmup
    for _ in range(max(1, N//10)):
      run_model(model_, s0, s1)

    st = time.time()

    tokenize_time = 0
    model_time = 0
    for _ in range(N):
      tt, mt, _ = run_model(model_, s0, s1)
      tokenize_time += tt
      model_time += mt
    # print(f"Execution time: {1000*(time.time() - st)/N:.2f} ms")
    print(f"Tokenize time: {(tokenize_time)/N:.2f} ms")
    print(f"Model time: {(model_time)/N:.2f} ms")


# s0 = "The company HuggingFace is based in New York City"
# s1 = "Apples are especially bad for your health"
# token_ids_mask = tokenizer.encode_plus(s0, s1, return_tensors="pt")

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased-finetuned-mrpc", torchscript=True)
model.eval()

if torch.cuda.is_available():
    model = model.to('cuda')

# Export with TorchScript
max_sequence_length = 64

dummy_input = [
        torch.zeros([1, max_sequence_length], dtype=torch.long, device=model.device),
        torch.zeros([1, max_sequence_length], dtype=torch.long, device=model.device),
        torch.ones([1, max_sequence_length], dtype=torch.long, device=model.device),
        ]


print('Traced Model')
traced_model = torch.jit.trace(model, dummy_input)
benchmark_model(traced_model, 10000)

print('Frozen Model')
frozen_model = torch.jit.optimize_for_inference(traced_model)
benchmark_model(frozen_model, 10000)