from contextlib import contextmanager
import os
import sys

import numpy as np

import torch
import torch.nn.functional as F

from .zoedepth.models.builder import build_model as build_model_internal
from .zoedepth.utils.config import get_config


@contextmanager
def silence_output(maybe=True):
    """
    This contextmanager will redirect stdout and stderr so that nothing is printed
    to the terminal. Taken from the link below:

    https://stackoverflow.com/questions/6735917/redirecting-stdout-to-nothing-in-python
    """
    if maybe:
        old_targets = sys.stdout, sys.stderr
        try:
            with open(os.devnull, "w") as new_target:
                sys.stdout = sys.stderr = new_target
                yield new_target, new_target
        finally:
            sys.stdout, sys.stderr = old_targets
    else:
        yield sys.stdout, sys.stderr


def build_model(ckpt=None, device=None, suppress_outputs=True):
    """
    Builds metric depth anything model from given checkpoint.
    Prepares model for eval on `device`.

    ckpt: Model checkpoint to load (defaults to model installed by `download_checkpoints.sh`)
    device: Device to put model on
    suppress_outputs: Stops depth-anything from outputting debug prints to stdout/err

    @return: Created metric depth anything model
    """
    if ckpt is None:
        ckpt = os.path.join(os.path.dirname(__file__), '../checkpoints/depth_anything_metric_depth_indoor.pt')

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    try:
        with silence_output(maybe=suppress_outputs):
            config = get_config("zoedepth", "eval", "nyu")
            config.pretrained_resource = f"local::{ckpt}"
            return build_model_internal(config).to(device).half().eval()
    except Exception:
        raise


def run_model(model, images, device=None):
    """
    Runs metric depth anything model on given images.

    model: Metric depth anything model (preferably created with `build_model`)
    images: Images to convert. Should be torch.Tensor (chw).
    device: Device to run on (defaults to device of model)

    @return: depths of given image(s), as torch.Tensor on `device` with shape (b, h, w)
    """
    if device is None:
        device = model.device
    else:
        model = model.to(device)

    h, w = images.shape[-2:]
    images = images.half().to(device)
    if len(images.shape) == 3:
        images = images.unsqueeze(dim=0)
        drop_batch_dim = True
    else:
        drop_batch_dim = False

    with torch.inference_mode():
        pred = model(images, dataset='nyu')
        if isinstance(pred, dict):
            pred = pred.get('metric_depth', pred.get('out'))
        elif isinstance(pred, (list, tuple)):
            pred = pred[-1]

        pred = F.interpolate(pred, (h, w), mode='bilinear', align_corners=False)[:, 0]  # drop channel dim
        depth = (pred[0] if drop_batch_dim else pred).detach()

    return depth
