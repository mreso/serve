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

venv_dir = 'venv'
package_path = "../models/bert_package.pt"

to_mock = [f'transformers.models.{s}.**' for s in os.listdir(f'./{venv_dir}/lib/python3.8/site-packages/transformers/models') if not (s.startswith('_') or s in ['bert', 'auto'])]

to_mock += [
  'transformers.pipelines.**',
  'tokenizers.tokenizers.**',
  'transformers.trainer*.**',
  'transformers.integrations.**',
  'transformers.optimization.**',
  'transformers.data.**',
  ]

with package.PackageExporter(package_path) as bert_package_exp:
    bert_package_exp.intern([
      'transformers.**',
      'packaging.**',
      'importlib_metadata.**',
      'tokenizers.**',
      'handler.**',
      ], exclude=to_mock )
    bert_package_exp.extern([
      'torch.**',
      'sys',
      'io',
      '__future__.**',
      '_queue',
    ])
    bert_package_exp.mock([
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
      ] + to_mock)

    bert_package_exp.save_pickle("handler", "handler.pkl", handler)


from torch import package
# Above code saves packaged resnet to disk.  To load, and execute:
bert_package_imp = package.PackageImporter(package_path)

handler_packaged = bert_package_imp.load_pickle("handler", "handler.pkl")


print(handler_packaged.execute(sequence_1, sequence_0))
print(handler_packaged.execute(sequence_2, sequence_0))
