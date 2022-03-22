import torch

print("This is handler creation")

class Handler(object):

    def __init__(self, ctx):
        self.state = ctx
        print("This is init")

    def handle(self, x):
        print(f"Handle this: {self.state}")
        return x

