import os.path as osp

import pytest
from ts.torch_handler.unit_tests.test_utils.mock_context import MockContext

from test_utils import REPO_ROOT


EXAMPLE_ROOT_DIR = osp.join(
    REPO_ROOT, "examples", "image_segmenter", "u2net"
)

def test_handler(monkeypatch, mocker):
    monkeypatch.syspath_prepend(EXAMPLE_ROOT_DIR)
    
    from object_segmentation_handler import ObjectSegmentationHandler

    handler = ObjectSegmentationHandler()
    ctx = MockContext(
        model_pt_file="u2net.pth",
        model_dir=osp.join(EXAMPLE_ROOT_DIR),
        model_file="model_factory.py",
    )

    handler.initialize(ctx)

    # Try empty string
    with open("/home/ubuntu/U-2-Net/test_data/test_images/corgy.jpg", "rb") as f:
        data = f.read()

    x = mocker.Mock(get=lambda _: data)

    x = handler.preprocess([x])
    x = handler.inference(x)
    x = handler.postprocess(x)
    return x
