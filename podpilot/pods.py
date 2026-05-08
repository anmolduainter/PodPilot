"""
Pod Lifecycle Management with rich progress display.
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import runpod
from rich.table import Table
from rich.live import Live
from rich.spinner import Spinner

from .console import console, retry


class PodStatus(str, Enum):
    RUNNING = "RUNNING"
    STOPPED = "EXITED"
    STARTING = "STARTING"
    UNKNOWN = "UNKNOWN"


@dataclass
class PodInfo:
    id: str
    name: str
    status: str
    gpu_type: str
    gpu_count: int
    image: str
    vram_gb: int
    vcpu: int
    memory_gb: int
    disk_gb: int
    volume_gb: int
    uptime_seconds: int
    ports: list
    env: dict
    cost_per_hour: float
    volume_mount_path: str

    @property
    def is_running(self) -> bool:
        return self.status == "RUNNING"

    @property
    def proxy_url(self) -> Optional[str]:
        if not self.is_running:
            return None
        for port in self.ports:
            if port.get("isIpPublic"):
                ip = port.get("ip", "")
                public_port = port.get("publicPort", "")
                return f"https://{ip}:{public_port}"
        return None


# ─── Retry-wrapped API calls ─────────────────────────────────────────────

@retry(max_attempts=3, base_delay=1.0)
def _api_get_pods():
    return runpod.get_pods()

@retry(max_attempts=3, base_delay=1.0)
def _api_get_pod(pod_id: str):
    return runpod.get_pod(pod_id)

@retry(max_attempts=3, base_delay=1.0)
def _api_create_pod(**kwargs):
    return runpod.create_pod(**kwargs)

@retry(max_attempts=3, base_delay=1.0)
def _api_stop_pod(pod_id: str):
    return runpod.stop_pod(pod_id)

@retry(max_attempts=3, base_delay=1.0)
def _api_resume_pod(pod_id: str, gpu_count: int = 1):
    return runpod.resume_pod(pod_id, gpu_count=gpu_count)

@retry(max_attempts=3, base_delay=1.0)
def _api_terminate_pod(pod_id: str):
    return runpod.terminate_pod(pod_id)


class PodManager:

    def get_all(self) -> list[PodInfo]:
        raw = _api_get_pods()
        pods_list = raw if isinstance(raw, list) else raw.get("pods", raw.get("myself", {}).get("pods", []))
        return [self._parse_pod(p) for p in pods_list]

    def get(self, pod_id: str) -> PodInfo:
        raw = _api_get_pod(pod_id)
        if isinstance(raw, dict) and "pod" in raw:
            raw = raw["pod"]
        return self._parse_pod(raw)

    def create(
        self,
        name: str,
        gpu_type_id: str,
        image: str = "",
        gpu_count: int = 1,
        disk_gb: int = 20,
        volume_gb: int = 0,
        ports: str = "8188/http",
        network_volume_id: str = None,
        env: dict = None,
        cloud_type: str = "ALL",
        docker_args: str = "",
        template_id: str = None,
    ) -> PodInfo:
        kwargs = dict(
            name=name,
            image_name=image,
            gpu_type_id=gpu_type_id,
            gpu_count=gpu_count,
            container_disk_in_gb=disk_gb,
            volume_in_gb=volume_gb,
            ports=ports,
            env=env or {},
            cloud_type=cloud_type,
            docker_args=docker_args,
        )
        if network_volume_id:
            kwargs["network_volume_id"] = network_volume_id
        if template_id:
            kwargs["template_id"] = template_id

        pod = _api_create_pod(**kwargs)
        pod_id = pod["id"]
        return self.get(pod_id)

    def stop(self, pod_id: str) -> PodInfo:
        _api_stop_pod(pod_id)
        time.sleep(2)
        return self.get(pod_id)

    def resume(self, pod_id: str, gpu_count: int = 1) -> PodInfo:
        _api_resume_pod(pod_id, gpu_count=gpu_count)
        return self.get(pod_id)

    def terminate(self, pod_id: str):
        _api_terminate_pod(pod_id)

    def wait_for_running(
        self,
        pod_id: str,
        timeout: int = 900,
        poll_interval: int = 10,
        on_timeout: str = "return",
    ) -> PodInfo:
        """
        Wait for pod to reach RUNNING status with live progress display.

        Args:
            pod_id: Pod to wait for
            timeout: Max wait time in seconds (default 900 = 15 min)
            poll_interval: Seconds between status checks
            on_timeout: "return" (default, returns pod in current state) or "raise" (raises TimeoutError)
        """
        start_time = time.time()
        deadline = start_time + timeout
        last_status = None

        with Live(console=console, refresh_per_second=4, transient=True) as live:
            while time.time() < deadline:
                elapsed = int(time.time() - start_time)

                try:
                    pod = self.get(pod_id)
                except Exception:
                    live.update(Spinner("dots", text=f"[yellow]  API hiccup, retrying... ({elapsed}s)[/yellow]"))
                    time.sleep(poll_interval)
                    continue

                current_status = pod.status

                if current_status != last_status:
                    console.print(f"  [dim]{elapsed:>4}s[/dim]  Status: [bold]{current_status}[/bold]")
                    last_status = current_status

                if pod.is_running and pod.uptime_seconds > 5:
                    return pod

                if current_status == "CREATED":
                    msg = "Provisioning GPU..."
                elif current_status in ("STARTING", "RESTARTING"):
                    msg = "Starting container, pulling image..."
                elif current_status == "RUNNING":
                    msg = "Container running, waiting for services..."
                else:
                    msg = f"Waiting ({current_status})..."

                progress = min(elapsed / timeout, 0.99)
                bar_len = 20
                filled = int(bar_len * progress)
                bar = f"[green]{'█' * filled}[/green][dim]{'░' * (bar_len - filled)}[/dim]"

                spinner_text = f"[cyan]{msg}[/cyan]  {bar}  [dim]{elapsed}s / {timeout}s[/dim]"
                live.update(Spinner("dots", text=spinner_text))
                time.sleep(poll_interval)

        # Timeout reached
        pod = self.get(pod_id)

        if on_timeout == "raise":
            raise TimeoutError(f"Pod {pod_id} not running after {timeout}s (last status: {pod.status})")

        console.print()
        console.print(f"[yellow]  Timeout after {timeout}s — pod status: [bold]{pod.status}[/bold][/yellow]")
        console.print(f"[yellow]  The pod may still be starting (large Docker image pull).[/yellow]")
        console.print(f"[yellow]  Pod ID: [bold]{pod.id}[/bold] — check with mgr.pod.get('{pod.id}')[/yellow]")
        return pod

    def find_by_name(self, name: str) -> Optional[PodInfo]:
        for pod in self.get_all():
            if pod.name == name:
                return pod
        return None

    def find_by_image(self, image: str) -> Optional[PodInfo]:
        for pod in self.get_all():
            if pod.image == image:
                return pod
        return None

    def print_all(self):
        pods = self.get_all()
        if not pods:
            console.print("[dim]No pods found.[/dim]")
            return

        table = Table(title="Pods", border_style="blue")
        table.add_column("", justify="center", width=3)
        table.add_column("Name", style="cyan", min_width=20)
        table.add_column("ID", style="dim")
        table.add_column("GPU")
        table.add_column("Disk", justify="right")
        table.add_column("Cost/hr", justify="right")
        table.add_column("State", justify="center")

        for pod in pods:
            icon = "[green]●[/green]" if pod.is_running else "[red]○[/red]"
            status_style = "green" if pod.is_running else "red"
            state = f"[{status_style}]{pod.status}[/{status_style}]"
            gpu = f"{pod.gpu_type}"
            cost = f"${pod.cost_per_hour:.2f}" if pod.cost_per_hour else "[dim]—[/dim]"
            disk = f"{pod.disk_gb}GB"
            table.add_row(icon, pod.name, pod.id, gpu, disk, cost, state)

        console.print(table)

    def _parse_pod(self, raw: dict) -> PodInfo:
        runtime = raw.get("runtime", {}) or {}
        machine = raw.get("machine", {}) or {}
        ports = runtime.get("ports", []) or []

        return PodInfo(
            id=raw.get("id", ""),
            name=raw.get("name", ""),
            status=raw.get("desiredStatus", "UNKNOWN"),
            gpu_type=machine.get("gpuDisplayName", raw.get("gpuTypeId", "?")),
            gpu_count=raw.get("gpuCount", 0),
            image=raw.get("imageName", ""),
            vram_gb=raw.get("memoryInGb", 0),
            vcpu=raw.get("vcpuCount", 0),
            memory_gb=raw.get("memoryInGb", 0),
            disk_gb=raw.get("containerDiskInGb", 0),
            volume_gb=raw.get("volumeInGb", 0),
            uptime_seconds=runtime.get("uptimeInSeconds", 0) if runtime else 0,
            ports=ports,
            env=raw.get("env", {}),
            cost_per_hour=raw.get("costPerHr", 0),
            volume_mount_path=raw.get("volumeMountPath", "/workspace"),
        )
