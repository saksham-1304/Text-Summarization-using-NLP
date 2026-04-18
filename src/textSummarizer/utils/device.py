"""Device detection and management utilities.

Provides centralized device detection logic to avoid code duplication
across model_trainer, model_evaluation, and prediction pipeline.
"""

from typing import Tuple

import torch

from textSummarizer.logging import logger


def get_device() -> torch.device:
    """Detect and log available compute device.
    
    Priority:
      1. CUDA (NVIDIA GPU)
      2. MPS (Apple Silicon)
      3. CPU (fallback)
    
    Returns:
        torch.device: Selected device (cuda, mps, or cpu).
    """
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"✓ Using CUDA: {device_name}")
        return torch.device("cuda")
    
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("✓ Using Apple MPS")
        return torch.device("mps")
    
    else:
        logger.warning("⚠ Using CPU (training will be slow)")
        return torch.device("cpu")


def get_device_and_dtype() -> Tuple[torch.device, torch.dtype]:
    """Get device and appropriate dtype for that device.
    
    Returns:
        Tuple[torch.device, torch.dtype]:
          - cuda -> (cuda, float16)
          - cpu -> (cpu, float32)  [float16 on CPU causes NaN]
          - mps -> (mps, float32)
    """
    device = get_device()
    
    if device.type == "cuda":
        dtype = torch.float16
        logger.info("Using float16 on CUDA for memory efficiency")
    else:
        dtype = torch.float32
        logger.info("Using float32 on CPU/MPS for numerical stability")
    
    return device, dtype
