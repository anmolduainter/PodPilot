"""
Shared utilities: Rich console, retry decorator, GPU composite scorer.
"""

import time
import functools
from typing import Optional

from rich.console import Console

console = Console()


def retry(max_attempts: int = 3, base_delay: float = 1.0, exceptions: tuple = (Exception,)):
    """Retry with exponential backoff for wrapping API calls."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    if attempt < max_attempts - 1:
                        delay = base_delay * (2 ** attempt)
                        console.print(f"[yellow]  Retry {attempt + 1}/{max_attempts - 1} after {delay:.1f}s: {e}[/yellow]")
                        time.sleep(delay)
            raise last_exc
        return wrapper
    return decorator


def compute_gpu_score(
    gpu,
    vram_needed: int,
    all_gpus: list = None,
    weights: dict = None,
) -> tuple[float, str]:
    """
    Composite GPU score combining fit, price, and availability.
    Returns (score, explanation_string).
    """
    w = weights or {"vram_fit": 0.3, "price": 0.4, "availability": 0.3}

    if all_gpus:
        max_vram = max(g.memory_gb for g in all_gpus) or 1
        prices = [g.best_price for g in all_gpus if g.best_price < float("inf")]
        max_price = max(prices) if prices else 1.0
    else:
        max_vram = gpu.memory_gb
        max_price = gpu.best_price if gpu.best_price < float("inf") else 1.0

    # vram_fit: prefer closest to needed (not wasteful overkill)
    excess = gpu.memory_gb - vram_needed
    vram_fit = max(0, 1.0 - (excess / max_vram)) if max_vram > 0 else 0.5

    # price: cheaper is better (normalized inverse)
    if gpu.best_price < float("inf") and max_price > 0:
        price_score = 1.0 - (gpu.best_price / max_price)
    else:
        price_score = 0.0

    # availability: prefer both clouds
    avail_score = gpu.availability_score / 3.0

    score = (
        w["vram_fit"] * vram_fit
        + w["price"] * price_score
        + w["availability"] * avail_score
    )

    # Build explanation
    reasons = []
    price_str = f"${gpu.best_price:.2f}/hr" if gpu.best_price < float("inf") else "N/A"
    reasons.append(f"{gpu.memory_gb}GB fits {vram_needed}GB need")
    reasons.append(price_str)
    clouds = []
    if gpu.secure_available:
        clouds.append("secure")
    if gpu.community_available:
        clouds.append("community")
    if clouds:
        reasons.append(f"available in {'+'.join(clouds)}")
    explanation = ", ".join(reasons)

    return score, explanation
