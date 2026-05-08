"""
Network Volume Management with smart data center + GPU cross-referencing.
Includes automated provisioning: create volume → spin up pod → run download script → terminate.
"""

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import runpod
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.spinner import Spinner

from .console import console, retry


@dataclass
class DataCenterInfo:
    id: str
    name: str
    location: str
    available_gpus: list

    @property
    def gpu_names(self) -> list[str]:
        return [g["gpuTypeId"].replace("NVIDIA ", "") for g in self.available_gpus if g.get("available")]

    def has_gpu_with_vram(self, min_vram: int, gpu_vram_map: dict) -> bool:
        for g in self.available_gpus:
            if not g.get("available"):
                continue
            vram = gpu_vram_map.get(g["gpuTypeId"], 0)
            if vram >= min_vram:
                return True
        return False

    def matching_gpus(self, min_vram: int, gpu_vram_map: dict) -> list[str]:
        result = []
        for g in self.available_gpus:
            if not g.get("available"):
                continue
            gpu_id = g["gpuTypeId"]
            vram = gpu_vram_map.get(gpu_id, 0)
            if vram < min_vram:
                continue
            name = gpu_id.replace("NVIDIA ", "")
            stock = g.get("stockStatus")
            stock_tag = " [Low]" if stock == "Low" else ""
            result.append(f"{name} ({vram}GB){stock_tag}")
        return result

    def availability_score(self, min_vram: int, gpu_vram_map: dict) -> float:
        score = 0.0
        for g in self.available_gpus:
            if not g.get("available"):
                continue
            vram = gpu_vram_map.get(g["gpuTypeId"], 0)
            if vram < min_vram:
                continue
            stock = g.get("stockStatus")
            score += 0.5 if stock == "Low" else 2.0
        return score


@dataclass
class VolumeInfo:
    id: str
    name: str
    size_gb: int
    data_center_id: str


@retry(max_attempts=3, base_delay=1.0)
def _query_graphql(query: str) -> dict:
    from runpod.api.graphql import run_graphql_query
    return run_graphql_query(query)


class VolumeManager:

    def __init__(self):
        self._dc_cache: list[DataCenterInfo] = []
        self._dc_cache_time: float = 0
        self._gpu_vram_map: dict = {}
        self._active_provision: dict = None

    # ─── Volume CRUD ──────────────────────────────────────────────────────

    def get_all(self) -> list[VolumeInfo]:
        result = _query_graphql('query { myself { networkVolumes { id name size dataCenterId } } }')
        volumes = result.get("data", {}).get("myself", {}).get("networkVolumes", [])
        return [VolumeInfo(id=v["id"], name=v["name"], size_gb=v["size"], data_center_id=v["dataCenterId"]) for v in volumes]

    def get(self, volume_id: str) -> Optional[VolumeInfo]:
        for vol in self.get_all():
            if vol.id == volume_id:
                return vol
        return None

    def find_by_name(self, name: str) -> Optional[VolumeInfo]:
        for vol in self.get_all():
            if vol.name == name:
                return vol
        return None

    def create(self, name: str, size_gb: int, data_center_id: str) -> VolumeInfo:
        query = f'''mutation {{ createNetworkVolume(input: {{ name: "{name}", size: {size_gb}, dataCenterId: "{data_center_id}" }}) {{ id name size dataCenterId }} }}'''
        result = _query_graphql(query)
        v = result["data"]["createNetworkVolume"]
        console.print(f"[green]Volume created:[/green] {v['name']} ({v['size']}GB) in {v['dataCenterId']}")
        return VolumeInfo(id=v["id"], name=v["name"], size_gb=v["size"], data_center_id=v["dataCenterId"])

    def resize(self, volume_id: str, new_size_gb: int) -> VolumeInfo:
        query = f'''mutation {{ updateNetworkVolume(input: {{ id: "{volume_id}", size: {new_size_gb} }}) {{ id name size dataCenterId }} }}'''
        result = _query_graphql(query)
        v = result["data"]["updateNetworkVolume"]
        console.print(f"[green]Volume resized:[/green] {v['name']} → {v['size']}GB")
        return VolumeInfo(id=v["id"], name=v["name"], size_gb=v["size"], data_center_id=v["dataCenterId"])

    def delete(self, volume_id: str):
        _query_graphql(f'''mutation {{ deleteNetworkVolume(input: {{ id: "{volume_id}" }}) }}''')
        console.print(f"[green]Volume deleted:[/green] {volume_id}")

    # ─── Data Center Discovery ────────────────────────────────────────────

    def get_data_centers(self, force_refresh: bool = False) -> list[DataCenterInfo]:
        if self._dc_cache and not force_refresh and (time.time() - self._dc_cache_time) < 300:
            return self._dc_cache
        query = '''query { dataCenters { id name location gpuAvailability { gpuTypeId available stockStatus } } }'''
        result = _query_graphql(query)
        dcs = result.get("data", {}).get("dataCenters", [])
        self._dc_cache = [
            DataCenterInfo(id=dc["id"], name=dc["name"], location=dc["location"], available_gpus=dc.get("gpuAvailability", []))
            for dc in dcs
        ]
        self._dc_cache_time = time.time()
        return self._dc_cache

    def _build_gpu_vram_map(self, gpu_manager) -> dict:
        if self._gpu_vram_map:
            return self._gpu_vram_map
        for gpu in gpu_manager.fetch_all():
            self._gpu_vram_map[gpu.id] = gpu.memory_gb
        return self._gpu_vram_map

    # ─── Smart Recommendations ────────────────────────────────────────────

    def recommend_data_center(self, min_vram: int, gpu_manager) -> list[DataCenterInfo]:
        vram_map = self._build_gpu_vram_map(gpu_manager)
        dcs = self.get_data_centers()
        matching = [dc for dc in dcs if dc.has_gpu_with_vram(min_vram, vram_map)]
        matching.sort(key=lambda dc: dc.availability_score(min_vram, vram_map), reverse=True)
        return matching

    def _get_gpus_in_dc_by_price(self, data_center_id: str, gpu_manager=None) -> list[tuple[float, str]]:
        """Get all available GPUs in a DC as [(price, gpu_id), ...] sorted cheapest first."""
        dcs = self.get_data_centers()
        dc_gpus = []
        for dc in dcs:
            if dc.id == data_center_id:
                dc_gpus = [g["gpuTypeId"] for g in dc.available_gpus if g.get("available")]
                break
        if not dc_gpus or not gpu_manager:
            return [(0.0, g) for g in dc_gpus]
        all_gpus = gpu_manager.fetch_all()
        priced = []
        for gid in dc_gpus:
            price = float("inf")
            for g in all_gpus:
                if g.id == gid:
                    price = g.best_price
                    break
            priced.append((price, gid))
        priced.sort(key=lambda x: x[0])
        return priced

    # ─── Provisioning ─────────────────────────────────────────────────────

    def provision(
        self,
        name: str,
        size_gb: int,
        download_script: str,
        gpu_vram: int,
        data_center_id: str = None,
        gpu_manager=None,
    ) -> VolumeInfo:
        """
        Full automated provisioning:
        1. Pick DC that has GPUs with gpu_vram GB+ VRAM
        2. Create network volume there
        3. Spin up CPU pod to download data (directly to /workspace)
        4. Terminate CPU pod when done

        Args:
            name: Volume name
            size_gb: Volume size in GB
            download_script: Path to local .sh file (downloads directly to /workspace)
            gpu_vram: Target GPU VRAM — places volume in a DC with these GPUs
            data_center_id: Explicit DC (skips auto-selection)
            gpu_manager: GPUManager instance for DC recommendations
        """
        script_path = Path(download_script)
        if not script_path.exists():
            raise FileNotFoundError(f"Script not found: {download_script}")
        script_content = script_path.read_text()

        # Step 1: Pick data center
        if not data_center_id:
            if gpu_vram > 0 and gpu_manager:
                recs = self.recommend_data_center(gpu_vram, gpu_manager)
                if not recs:
                    raise RuntimeError(f"No DC with {gpu_vram}GB+ GPUs")
                data_center_id = recs[0].id
                console.print(f"  [green]Best DC:[/green] {data_center_id} ({recs[0].location})")
            else:
                data_center_id = "US-TX-3"

        # Step 2: Create or reuse volume
        existing = self.find_by_name(name)
        if existing:
            vol = existing
            data_center_id = vol.data_center_id
            console.print(f"  [green]Volume '{name}' already exists:[/green] {vol.id} in {data_center_id}")
        else:
            console.print(Panel(
                f"[bold]Name:[/bold]   {name}\n[bold]Size:[/bold]   {size_gb}GB\n[bold]DC:[/bold]     {data_center_id}",
                title="Creating Network Volume", border_style="blue"
            ))
            vol = self.create(name, size_gb, data_center_id)

        # Step 3: Get cheapest GPU pod (under $1/hr) in same DC for fast download
        gpu_options = self._get_gpus_in_dc_by_price(data_center_id, gpu_manager)
        budget_gpus = [(p, g) for p, g in gpu_options if p < 1.0]
        if not budget_gpus:
            budget_gpus = gpu_options[:3]

        if budget_gpus:
            names = ", ".join(f"{g.replace('NVIDIA ', '')} (${p:.2f})" for p, g in budget_gpus[:5])
            console.print(f"  [dim]GPUs under $1/hr: {names}[/dim]")

        pod = None
        for price, gpu_id in budget_gpus:
            gpu_short = gpu_id.replace("NVIDIA ", "")
            try:
                console.print(f"  [cyan]Trying {gpu_short} (${price:.2f}/hr)...[/cyan]")
                pod = runpod.create_pod(
                    name=f"{name}-setup",
                    image_name="runpod/base:0.4.0-cuda11.8.0",
                    gpu_type_id=gpu_id,
                    gpu_count=1,
                    container_disk_in_gb=60,
                    volume_in_gb=0,
                    ports="22/tcp",
                    network_volume_id=vol.id,
                    data_center_id=data_center_id,
                )
                console.print(f"  [green]Got {gpu_short}![/green]")
                break
            except Exception:
                console.print(f"  [yellow]{gpu_short} not available.[/yellow]")
                continue

        if not pod:
            console.print(f"  [red]No GPU under $1/hr available in {data_center_id}.[/red]")
            console.print(f"  [yellow]Volume {vol.name} ({vol.id}) created but empty.[/yellow]")
            return vol

        pod_id = pod["id"]
        console.print(f"  [green]Pod created:[/green] {pod_id}")

        try:
            # Step 5: Wait for SSH
            ssh_host, ssh_port = self._wait_for_ssh(pod_id)
            console.print(f"  [green]SSH ready:[/green] {ssh_host}:{ssh_port}")

            self._active_provision = {
                "pod_id": pod_id, "ssh_host": ssh_host, "ssh_port": ssh_port,
                "volume_id": vol.id, "volume_name": name,
            }

            # Step 6: Upload and run script
            console.print(f"  [cyan]Running download script...[/cyan]")
            console.print(f"  [dim]If disconnected, check: mgr.check_provision()[/dim]")
            console.print()
            self._run_script_via_ssh(ssh_host, ssh_port, script_content)
            console.print(f"\n  [green bold]Download complete![/green bold]")

        except KeyboardInterrupt:
            console.print(f"\n  [yellow]Disconnected. Download continues on the pod.[/yellow]")
            console.print(f"  [yellow]Check: mgr.check_provision()[/yellow]")
            console.print(f"  [yellow]When done: mgr.finish_provision()[/yellow]")
            return vol
        except Exception as e:
            console.print(f"  [red]Error: {e}[/red]")
            console.print(f"  [yellow]Pod {pod_id} left running. Check: mgr.check_provision()[/yellow]")
            return vol
        else:
            console.print(f"  [dim]Terminating download pod...[/dim]")
            runpod.terminate_pod(pod_id)
            self._active_provision = None

        console.print(Panel(
            f"[bold]Volume:[/bold]  {vol.name} ({vol.size_gb}GB)\n"
            f"[bold]ID:[/bold]      {vol.id}\n"
            f"[bold]DC:[/bold]      {vol.data_center_id}\n"
            f"[bold]Status:[/bold]  Ready to use",
            title="Volume Provisioned", border_style="green"
        ))
        return vol

    # ─── SSH Helpers ──────────────────────────────────────────────────────

    def _wait_for_ssh(self, pod_id: str, timeout: int = 300) -> tuple[str, int]:
        deadline = time.time() + timeout
        with Live(console=console, refresh_per_second=2, transient=True) as live:
            while time.time() < deadline:
                elapsed = int(time.time() - (deadline - timeout))
                try:
                    pod_info = runpod.get_pod(pod_id)
                    runtime = pod_info.get("runtime", {}) or {}
                    for p in (runtime.get("ports", []) or []):
                        if p.get("privatePort") == 22 and p.get("isIpPublic"):
                            # Wait for SSH daemon to fully initialize
                            console.print(f"  [dim]SSH port found, waiting for daemon...[/dim]")
                            time.sleep(20)
                            # Verify SSH is actually accepting connections
                            test = subprocess.run(
                                f"ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=5 -p {p['publicPort']} root@{p['ip']} echo ok",
                                shell=True, capture_output=True, text=True, timeout=15
                            )
                            if test.returncode == 0:
                                return p["ip"], p["publicPort"]
                            console.print(f"  [dim]SSH not ready yet, retrying...[/dim]")
                except Exception:
                    pass
                live.update(Spinner("dots", text=f"[cyan]Waiting for SSH... ({elapsed}s)[/cyan]"))
                time.sleep(10)
        raise TimeoutError(f"SSH not ready after {timeout}s")

    def _run_script_via_ssh(self, host: str, port: int, script_content: str):
        import tempfile
        wrapper = self._build_status_wrapper(script_content)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write(wrapper)
            local_script = f.name

        ssh_opts = "-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"

        scp_cmd = f"scp {ssh_opts} -P {port} {local_script} root@{host}:/tmp/setup.sh"
        result = subprocess.run(scp_cmd, shell=True, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f"SCP failed: {result.stderr}")

        ssh_cmd = f"ssh {ssh_opts} -p {port} root@{host} 'bash /tmp/setup.sh'"
        process = subprocess.Popen(ssh_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

        try:
            for line in process.stdout:
                line = line.rstrip()
                if not line:
                    continue
                if line == "___PROVISION_COMPLETE___":
                    break
                if line == "___PROVISION_FAILED___":
                    process.kill()
                    raise RuntimeError("Download script failed")
                if "%" in line and ("K" in line or "M" in line or "G" in line):
                    console.print(f"  [dim]│[/dim] [cyan]{line}[/cyan]")
                elif line.startswith("==="):
                    console.print(f"  [dim]│[/dim] [green bold]{line}[/green bold]")
                elif line.startswith("Copying") or line.startswith("Done"):
                    console.print(f"  [dim]│[/dim] [green]{line}[/green]")
                else:
                    console.print(f"  [dim]│[/dim] {line}")

            process.wait(timeout=3600)
            if process.returncode != 0:
                raise RuntimeError(f"Script exited with code {process.returncode}")
        except KeyboardInterrupt:
            console.print("\n  [yellow]Disconnected. Script may still be running on the pod.[/yellow]")
            console.print(f"  [yellow]Check pod status and terminate manually when done.[/yellow]")
        finally:
            try:
                process.kill()
            except Exception:
                pass
        Path(local_script).unlink(missing_ok=True)

    def _build_status_wrapper(self, script_content: str) -> str:
        return f'''#!/bin/bash
set -e
trap 'echo "___PROVISION_FAILED___"' ERR
{script_content}
echo "___PROVISION_COMPLETE___"
'''

    def provision_status(self, host: str, port: int) -> dict:
        import json as _json
        ssh_opts = "-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
        result = subprocess.run(
            f"ssh {ssh_opts} -p {port} root@{host} 'cat /workspace/.provision_status.json 2>/dev/null; echo; tail -5 /workspace/.provision.log 2>/dev/null'",
            shell=True, capture_output=True, text=True, timeout=15
        )
        if result.returncode != 0:
            return {"state": "unknown", "error": "Could not connect"}
        lines = result.stdout.strip().split("\n")
        try:
            status = _json.loads(lines[0])
        except Exception:
            status = {"state": "unknown"}
        status["recent_logs"] = lines[1:] if len(lines) > 1 else []
        return status

    def provision_logs(self, host: str, port: int, lines: int = 30) -> str:
        ssh_opts = "-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR"
        result = subprocess.run(
            f"ssh {ssh_opts} -p {port} root@{host} 'tail -{lines} /workspace/.provision.log 2>/dev/null'",
            shell=True, capture_output=True, text=True, timeout=15
        )
        if result.returncode != 0:
            return "Could not connect"
        console.print(Panel(result.stdout, title="Provision Logs", border_style="blue"))
        return result.stdout

    # ─── Display ──────────────────────────────────────────────────────────

    def print_all(self):
        volumes = self.get_all()
        if not volumes:
            console.print("[dim]No network volumes found.[/dim]")
            return
        table = Table(title="Network Volumes", border_style="blue")
        table.add_column("Name", style="cyan")
        table.add_column("ID", style="dim")
        table.add_column("Size", justify="right", style="bold")
        table.add_column("Data Center", justify="right")
        for vol in volumes:
            table.add_row(vol.name, vol.id, f"{vol.size_gb}GB", vol.data_center_id)
        console.print(table)

    def print_recommendations(self, min_vram: int, gpu_manager):
        vram_map = self._build_gpu_vram_map(gpu_manager)
        recs = self.recommend_data_center(min_vram, gpu_manager)
        if not recs:
            console.print(f"[red]No data centers with {min_vram}GB+ GPUs available.[/red]")
            return
        table = Table(title=f"Best Data Centers for {min_vram}GB+ VRAM", border_style="green")
        table.add_column("#", justify="right", style="dim", width=3)
        table.add_column("Data Center", style="cyan")
        table.add_column("Location")
        table.add_column("Score", justify="center", style="green bold")
        table.add_column("Available GPUs")
        for i, dc in enumerate(recs[:10], 1):
            gpus = dc.matching_gpus(min_vram, vram_map)
            score = dc.availability_score(min_vram, vram_map)
            table.add_row(str(i), dc.id, dc.location, f"{score:.1f}", ", ".join(gpus))
        console.print(table)
        console.print("[dim]  Score: good stock = 2pts/GPU, low stock = 0.5pts/GPU. [Low] = limited availability.[/dim]")
