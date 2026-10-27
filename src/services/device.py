"""Shared CPU/CUDA selection for local inference."""
import logging


def inference_device(use_gpu):
    import torch
    if use_gpu and torch.cuda.is_available():
        return 'cuda'
    if use_gpu:
        logging.getLogger(__name__).warning('CUDA unavailable; using CPU')
    return 'cpu'
