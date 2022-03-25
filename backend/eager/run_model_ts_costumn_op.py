import torch

def load_and_execute():
    model = torch.jit.load('module_with_tokenizer.pt')

    print(model('Apples are great for your health', 'Newton was struck by an apple'))

try:
    load_and_execute()
except RuntimeError:
    print('First try fails due to unloaded op')

torch.ops.load_library('libbert_tokenizer.so')
load_and_execute()
