"""
GPU Discovery, Filtering, and Smart Selection.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import runpod
from rich.table import Table

from .console import console, retry, compute_gpu_score


class CloudType(str, Enum):
    ALL = "ALL"
    SECURE = "SECURE"
    COMMUNITY = "COMMUNITY"


class SortBy(str, Enum):
    PRICE = "price"
    VRAM = "vram"
    AVAILABILITY = "availability"
    VRAM_PER_DOLLAR = "vram_per_dollar"
    SMART = "smart"


@dataclass
class GPUInfo:
    id: str
    display_name: str
    manufacturer: str
    memory_gb: int
    cuda_cores: int
    max_gpu_count: int
    secure_available: bool
    community_available: bool
    secure_price: Optional[float] = None
    community_price: Optional[float] = None
    spot_price: Optional[float] = None
    lowest_price: Optional[float] = None
    one_month_price: Optional[float] = None
    three_month_price: Optional[float] = None

    @property
    def available(self) -> bool:
        return self.secure_available or self.community_available

    @property
    def best_price(self) -> float:
        prices = [p for p in [self.lowest_price, self.spot_price, self.community_price, self.secure_price] if p]
        return min(prices) if prices else float("inf")

    @property
    def vram_per_dollar(self) -> float:
        price = self.best_price
        if price and price > 0:
            return self.memory_gb / price
        return 0.0

    @property
    def availability_score(self) -> int:
        score = 0
        if self.secure_available:
            score += 2
        if self.community_available:
            score += 1
        return score

    def summary(self) -> str:
        status = "AVAILABLE" if self.available else "unavailable"
        price_str = f"${self.best_price:.2f}/hr" if self.best_price < float("inf") else "N/A"
        return f"{self.display_name:<30} {self.memory_gb:>3}GB  {price_str:>10}  [{status}]"


@retry(max_attempts=3, base_delay=1.0)
def _fetch_gpu_list():
    return runpod.get_gpus()


@retry(max_attempts=3, base_delay=1.0)
def _fetch_gpu_detail(gpu_id: str):
    return runpod.get_gpu(gpu_id)


class GPUManager:

    def __init__(self):
        self._cache: list[GPUInfo] = []
        self._cache_time: float = 0

    def fetch_all(self, force_refresh: bool = False) -> list[GPUInfo]:
        """Fetch all GPU types with full pricing and availability."""
        import time as _time
        if self._cache and not force_refresh and (_time.time() - self._cache_time) < 300:
            return self._cache

        gpu_list = _fetch_gpu_list()
        gpus = []
        for basic in gpu_list:
            gpu_id = basic.get("id", "")
            if not gpu_id:
                continue
            try:
                g = _fetch_gpu_detail(gpu_id)
            except Exception:
                console.print(f"[yellow]  Skipped detail fetch for {gpu_id}[/yellow]")
                g = basic

            gpu = GPUInfo(
                id=g.get("id", gpu_id),
                display_name=g.get("displayName", ""),
                manufacturer=g.get("manufacturer", ""),
                memory_gb=g.get("memoryInGb", 0),
                cuda_cores=g.get("cudaCores", 0),
                max_gpu_count=g.get("maxGpuCount", 0),
                secure_available=g.get("secureCloud", False),
                community_available=g.get("communityCloud", False),
                secure_price=_safe_float(g.get("securePrice")),
                community_price=_safe_float(g.get("communityPrice")),
                spot_price=_safe_float(g.get("communitySpotPrice") or g.get("secureSpotPrice")),
                lowest_price=_safe_float(g.get("lowestPrice", {}).get("minimumBidPrice") if isinstance(g.get("lowestPrice"), dict) else g.get("lowestPrice")),
                one_month_price=_safe_float(g.get("oneMonthPrice")),
                three_month_price=_safe_float(g.get("threeMonthPrice")),
            )
            gpus.append(gpu)

        self._cache = gpus
        self._cache_time = _time.time()
        return gpus

    def filter(
        self,
        min_vram: int = 0,
        max_vram: int = 999,
        available_only: bool = False,
        cloud_type: CloudType = CloudType.ALL,
        manufacturer: str = None,
        max_price: float = None,
    ) -> list[GPUInfo]:
        """Filter GPUs by criteria."""
        gpus = self.fetch_all()
        results = []
        for gpu in gpus:
            if gpu.memory_gb < min_vram or gpu.memory_gb > max_vram:
                continue
            if available_only and not gpu.available:
                continue
            if cloud_type == CloudType.SECURE and not gpu.secure_available:
                continue
            if cloud_type == CloudType.COMMUNITY and not gpu.community_available:
                continue
            if manufacturer and manufacturer.lower() not in gpu.manufacturer.lower():
                continue
            if max_price and gpu.best_price > max_price:
                continue
            results.append(gpu)
        return results

    def sort(self, gpus: list[GPUInfo] = None, by: SortBy = SortBy.PRICE, descending: bool = False, vram_needed: int = 0) -> list[GPUInfo]:
        """Sort GPUs by criteria."""
        if gpus is None:
            gpus = self.fetch_all()

        if by == SortBy.SMART:
            scored = [(compute_gpu_score(g, vram_needed=vram_needed, all_gpus=gpus)[0], g) for g in gpus]
            scored.sort(key=lambda x: x[0], reverse=True)
            return [g for _, g in scored]

        key_map = {
            SortBy.PRICE: lambda g: g.best_price,
            SortBy.VRAM: lambda g: g.memory_gb,
            SortBy.AVAILABILITY: lambda g: g.availability_score,
            SortBy.VRAM_PER_DOLLAR: lambda g: g.vram_per_dollar,
        }
        return sorted(gpus, key=key_map[by], reverse=descending)

    def find_best(
        self,
        min_vram: int = 0,
        max_price: float = None,
        prefer: SortBy = SortBy.PRICE,
    ) -> Optional[GPUInfo]:
        """Find the best available GPU matching requirements."""
        candidates = self.filter(min_vram=min_vram, available_only=True, max_price=max_price)
        if not candidates:
            return None

        if prefer == SortBy.SMART:
            sorted_gpus = self.sort(candidates, by=SortBy.SMART, vram_needed=min_vram)
        else:
            sorted_gpus = self.sort(candidates, by=prefer, descending=(prefer in [SortBy.VRAM, SortBy.VRAM_PER_DOLLAR, SortBy.AVAILABILITY]))
        return sorted_gpus[0]

    def recommend(self, vram_needed: int, budget_per_hour: float = None) -> list[GPUInfo]:
        """Smart recommendation sorted by composite score."""
        candidates = self.filter(min_vram=vram_needed, available_only=True, max_price=budget_per_hour)
        return self.sort(candidates, by=SortBy.SMART, vram_needed=vram_needed)

    def print_table(self, gpus: list[GPUInfo] = None):
        """Print a rich formatted table of GPUs."""
        if gpus is None:
            gpus = self.fetch_all()

        table = Table(title="GPU Availability", border_style="blue", show_lines=False)
        table.add_column("GPU", style="cyan", min_width=25)
        table.add_column("VRAM", justify="right", style="bold")
        table.add_column("Price/hr", justify="right")
        table.add_column("Spot", justify="right", style="dim")
        table.add_column("Avail", justify="center")
        table.add_column("Cloud", justify="right")

        for gpu in gpus:
            price = f"${gpu.best_price:.2f}" if gpu.best_price < float("inf") else "[dim]N/A[/dim]"
            spot = f"${gpu.spot_price:.2f}" if gpu.spot_price else "[dim]—[/dim]"
            avail = "[green bold]YES[/green bold]" if gpu.available else "[red]no[/red]"
            cloud = []
            if gpu.secure_available:
                cloud.append("[blue]secure[/blue]")
            if gpu.community_available:
                cloud.append("[magenta]community[/magenta]")
            cloud_str = "+".join(cloud) if cloud else "[dim]none[/dim]"
            table.add_row(gpu.display_name, f"{gpu.memory_gb}GB", price, spot, avail, cloud_str)

        console.print(table)


def _safe_float(val) -> Optional[float]:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None
