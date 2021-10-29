#!/usr/bin/env python
# coding: utf-8

import os
from unittest.mock import patch

import torch
from torch import package

from transformers import AutoTokenizer, AutoModelForSequenceClassification

sequence_0 = "The company HuggingFace is based in New York City"
sequence_1 = "Apples are especially bad for your health"
sequence_2 = "HuggingFace's headquarters are situated in Manhattan"

from handler import Handler

handler = Handler()

print(handler.execute(sequence_0, sequence_1))
print(handler.execute(sequence_0, sequence_2))

package_path = '../models/bert_external_transformers.pt'

with package.PackageExporter(package_path) as bert_package_exp:
    bert_package_exp.intern([
      'handler.**',
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

    bert_package_exp.save_pickle("handler", "handler.pkl", handler)


from torch import package
# Above code saves packaged resnet to disk.  To load, and execute:
bert_package_imp = package.PackageImporter(package_path)

handler_packaged = bert_package_imp.load_pickle("handler", "handler.pkl")

print(handler_packaged.execute(sequence_0, sequence_1))
print(handler_packaged.execute(sequence_0, sequence_2))
