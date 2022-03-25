import torch

torch.ops.load_library('libbert_tokenizer.so')

class MyModel(torch.nn.Module):
    def __init__(self):
        self.tokenize = torch.ops.bert.tokenize
        super().__init__()

    def forward(self, s1: str, s2: str):
        return self.tokenize(s1, s2, 64)


model = MyModel()

print(model('Apples are great for your health', 'Newton was struck by an apple'))