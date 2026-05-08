"""
RunPodManager — Smart orchestrator with rich output.
"""

import os
from typing import Optional

import runpod
from rich.panel import Panel
from rich.text import Text
from rich.table import Table

from .gpu import GPUManager, GPUInfo, SortBy, CloudType
from .pods import PodManager, PodInfo
from .volumes import VolumeManager, VolumeInfo
from .console import console, compute_gpu_score


class RunPodManager:
    """
    Smart RunPod infrastructure manager.

    Usage:
        mgr = RunPodManager(api_key="...")

        mgr.gpus()                                    # Rich table of all GPUs
        mgr.gpus(min_vram=48, available_only=True)    # Filtered
        mgr.best_gpu(min_vram=48)                     # Smart auto-pick
        mgr.recommend(vram_needed=48, budget=1.5)     # Top recommendations

        pod = mgr.launch("my-pod", image="my/image", min_vram=48)
        mgr.pods()
        mgr.stop(name="my-pod")
        mgr.resume(name="my-pod")
        mgr.terminate(name="my-pod")

        mgr.balance()
        mgr.status()
    """

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("RUNPOD_API_KEY")
        if not self.api_key:
            console.print("[red bold]Error:[/red bold] Provide api_key or set RUNPOD_API_KEY env var")
            raise ValueError("Provide api_key or set RUNPOD_API_KEY env var")
        runpod.api_key = self.api_key
        self.gpu = GPUManager()
        self.pod = PodManager()
        self.volume = VolumeManager()

    # ─── GPU Discovery ────────────────────────────────────────────────────

    def gpus(
        self,
        min_vram: int = 0,
        max_vram: int = 999,
        available_only: bool = False,
        max_price: float = None,
        sort_by: SortBy = SortBy.VRAM,
        cloud_type: CloudType = CloudType.ALL,
        show: bool = True,
    ) -> list[GPUInfo]:
        """List GPUs with optional filtering and sorting."""
        filtered = self.gpu.filter(
            min_vram=min_vram,
            max_vram=max_vram,
            available_only=available_only,
            cloud_type=cloud_type,
            max_price=max_price,
        )
        desc = sort_by in [SortBy.VRAM, SortBy.VRAM_PER_DOLLAR, SortBy.AVAILABILITY]
        sorted_gpus = self.gpu.sort(filtered, by=sort_by, descending=desc, vram_needed=min_vram)
        if show:
            self.gpu.print_table(sorted_gpus)
        return sorted_gpus

    def best_gpu(
        self,
        min_vram: int = 0,
        max_price: float = None,
        prefer: SortBy = SortBy.SMART,
    ) -> Optional[GPUInfo]:
        """Auto-select the best available GPU with explanation."""
        gpu = self.gpu.find_best(min_vram=min_vram, max_price=max_price, prefer=prefer)
        if gpu:
            score, explanation = compute_gpu_score(
                gpu, vram_needed=min_vram,
                all_gpus=self.gpu.filter(min_vram=min_vram, available_only=True, max_price=max_price),
            )
            content = Text()
            content.append(f"{gpu.display_name}", style="cyan bold")
            content.append(f"  {gpu.memory_gb}GB  ")
            price_str = f"${gpu.best_price:.2f}/hr" if gpu.best_price < float("inf") else "N/A"
            content.append(price_str, style="green")
            content.append(f"\n{explanation}", style="dim")
            console.print(Panel(content, title="Best GPU", border_style="green"))
        else:
            msg = f"No available GPU with {min_vram}GB+ VRAM"
            if max_price:
                msg += f" under ${max_price}/hr"
            console.print(Panel(msg, title="No GPU Found", border_style="red"))
        return gpu

    def recommend(self, vram_needed: int, budget: float = None) -> list[GPUInfo]:
        """Smart GPU recommendations sorted by composite score."""
        recs = self.gpu.recommend(vram_needed, budget)
        if recs:
            table = Table(title=f"Recommendations for {vram_needed}GB VRAM", border_style="green")
            table.add_column("#", justify="right", style="dim", width=3)
            table.add_column("GPU", style="cyan", min_width=25)
            table.add_column("VRAM", justify="right", style="bold")
            table.add_column("Price/hr", justify="right")
            table.add_column("Score", justify="right", style="green")
            table.add_column("Why", style="dim")

            for i, gpu in enumerate(recs[:5], 1):
                score, explanation = compute_gpu_score(gpu, vram_needed=vram_needed, all_gpus=recs)
                price = f"${gpu.best_price:.2f}" if gpu.best_price < float("inf") else "N/A"
                table.add_row(str(i), gpu.display_name, f"{gpu.memory_gb}GB", price, f"{score:.2f}", explanation)
            console.print(table)
        else:
            console.print(f"[red]No GPUs available with {vram_needed}GB+ VRAM[/red]")
        return recs

    # ─── Network Volumes ────────────────────────────────────────────────

    def volumes(self, show: bool = True) -> list[VolumeInfo]:
        """List all network volumes."""
        vols = self.volume.get_all()
        if show:
            self.volume.print_all()
        return vols

    def create_volume(self, name: str, size_gb: int, data_center_id: str = None, min_vram: int = 0) -> VolumeInfo:
        """
        Create a network volume. Auto-picks best data center if not specified.

        Args:
            name: Volume name
            size_gb: Size in GB
            data_center_id: Explicit data center (e.g. "US-TX-3")
            min_vram: If data_center_id not given, pick a DC with this much GPU VRAM available

        Examples:
            mgr.create_volume("my-models", 60, data_center_id="US-TX-3")
            mgr.create_volume("my-models", 60, min_vram=48)  # auto-picks DC with 48GB GPUs
        """
        if not data_center_id:
            if min_vram > 0:
                recs = self.volume.recommend_data_center(min_vram, self.gpu)
                if not recs:
                    console.print(f"[red]No data center with {min_vram}GB+ GPUs available.[/red]")
                    raise RuntimeError(f"No data center with {min_vram}GB+ GPUs")
                data_center_id = recs[0].id
                console.print(f"  [green]Auto-selected DC:[/green] {data_center_id} ({recs[0].location})")
            else:
                data_center_id = "US-TX-3"
                console.print(f"  [dim]Using default DC: {data_center_id}[/dim]")

        return self.volume.create(name, size_gb, data_center_id)

    def delete_volume(self, volume_id: str = None, name: str = None):
        """Delete a network volume by ID or name."""
        if not volume_id and name:
            vol = self.volume.find_by_name(name)
            if not vol:
                console.print(f"[red]No volume found with name '{name}'[/red]")
                raise ValueError(f"No volume found with name '{name}'")
            volume_id = vol.id
        self.volume.delete(volume_id)

    def resize_volume(self, volume_id: str = None, name: str = None, size_gb: int = 0) -> VolumeInfo:
        """Resize a network volume (can only increase)."""
        if not volume_id and name:
            vol = self.volume.find_by_name(name)
            if not vol:
                raise ValueError(f"No volume found with name '{name}'")
            volume_id = vol.id
        return self.volume.resize(volume_id, size_gb)

    def provision_volume(
        self,
        name: str,
        size_gb: int,
        download_script: str,
        gpu_vram: int,
        data_center_id: str = None,
    ) -> VolumeInfo:
        """
        One-command volume provisioning:
        1. Pick data center that has GPUs with `gpu_vram`GB+ VRAM
        2. Create network volume there
        3. Spin up CPU pod to download data (runs your .sh script)
        4. Terminate CPU pod when done

        The volume ends up in a DC where your target GPU is available,
        so you can attach it to a GPU pod later without region mismatch.

        Args:
            name: Volume name
            size_gb: Volume size in GB
            download_script: Path to .sh file (downloads directly to /workspace)
            gpu_vram: Target GPU VRAM in GB — places volume in a DC with these GPUs
            data_center_id: Explicit data center (skips auto-pick)

        Example:
            vol = mgr.provision_volume(
                "ltx23-models", size_gb=60,
                download_script="download_models.sh",
                gpu_vram=48,  # volume goes where 48GB GPUs exist
            )
        """
        return self.volume.provision(
            name=name,
            size_gb=size_gb,
            download_script=download_script,
            gpu_vram=gpu_vram,
            data_center_id=data_center_id,
            gpu_manager=self.gpu,
        )

    def check_provision(self) -> dict:
        """Check status of an active volume provisioning (download in progress)."""
        prov = self.volume._active_provision
        if not prov:
            console.print("[dim]No active provisioning.[/dim]")
            return {}

        console.print(f"  [cyan]Checking pod {prov['pod_id']}...[/cyan]")
        status = self.volume.provision_status(prov["ssh_host"], prov["ssh_port"])

        state = status.get("state", "unknown")
        desc = status.get("description", "")
        step = status.get("step", "?")
        total = status.get("total", "?")

        state_style = {"running": "cyan", "completed": "green", "failed": "red"}.get(state, "yellow")
        console.print(Panel(
            f"[bold]Volume:[/bold]  {prov['volume_name']}\n"
            f"[bold]Pod:[/bold]     {prov['pod_id']}\n"
            f"[bold]State:[/bold]   [{state_style}]{state}[/{state_style}]\n"
            f"[bold]Step:[/bold]    {step}/{total} — {desc}",
            title="Provision Status", border_style=state_style
        ))

        if status.get("recent_logs"):
            console.print("  [dim]Recent logs:[/dim]")
            for log in status["recent_logs"]:
                if log.strip():
                    console.print(f"  [dim]│[/dim] {log}")

        return status

    def provision_logs(self, lines: int = 30) -> str:
        """Show recent download logs from active provisioning."""
        prov = self.volume._active_provision
        if not prov:
            console.print("[dim]No active provisioning.[/dim]")
            return ""
        return self.volume.provision_logs(prov["ssh_host"], prov["ssh_port"], lines)

    def finish_provision(self):
        """Terminate the download pod after provisioning is done."""
        prov = self.volume._active_provision
        if not prov:
            console.print("[dim]No active provisioning.[/dim]")
            return
        console.print(f"  [dim]Terminating download pod {prov['pod_id']}...[/dim]")
        runpod.terminate_pod(prov["pod_id"])
        self.volume._active_provision = None
        console.print(f"  [green]Done. Volume {prov['volume_name']} ready.[/green]")

    def recommend_volume_location(self, min_vram: int, show: bool = True) -> list:
        """Show best data centers for creating a volume that has GPUs with enough VRAM."""
        if show:
            self.volume.print_recommendations(min_vram, self.gpu)
        return self.volume.recommend_data_center(min_vram, self.gpu)

    # ─── Pod Lifecycle ────────────────────────────────────────────────────

    def pods(self, show: bool = True) -> list[PodInfo]:
        all_pods = self.pod.get_all()
        if show:
            self.pod.print_all()
        return all_pods

    def launch(
        self,
        name: str,
        image: str = None,
        gpu: str = None,
        min_vram: int = 0,
        max_price: float = None,
        prefer: SortBy = SortBy.SMART,
        gpu_count: int = 1,
        container_disk_gb: int = 20,
        volume_gb: int = 0,
        ports: str = "8188/http",
        network_volume_id: str = None,
        env: dict = None,
        cloud_type: str = "ALL",
        docker_args: str = "",
        template_id: str = None,
        wait: bool = True,
        timeout: int = 900,
    ) -> PodInfo:
        """
        Smart pod creation with auto GPU selection and robust startup.

        Args:
            name: Pod name
            image: Docker image
            gpu: Explicit GPU type ID (skips auto-selection)
            min_vram: Min VRAM for auto-selection
            max_price: Max price/hr for auto-selection
            prefer: GPU selection strategy (SMART, PRICE, VRAM, VRAM_PER_DOLLAR)
            gpu_count: Number of GPUs
            container_disk_gb: Ephemeral container storage
            volume_gb: Persistent workspace volume
            ports: Exposed ports
            network_volume_id: Network volume to attach
            env: Environment variables
            cloud_type: ALL, SECURE, or COMMUNITY
            docker_args: Extra docker arguments
            template_id: RunPod template ID
            wait: Wait for pod to be ready
            timeout: Max wait time in seconds (default 900 = 15 min)
        """
        existing = self.pod.find_by_name(name)
        if existing:
            if existing.is_running:
                console.print(f"[green]Pod '{name}' already running:[/green] {existing.id}")
                return existing
            else:
                console.print(f"[yellow]Pod '{name}' exists but stopped. Resuming...[/yellow]")
                self.pod.resume(existing.id, gpu_count=gpu_count)
                if wait:
                    return self.pod.wait_for_running(existing.id, timeout=timeout, on_timeout="return")
                return self.pod.get(existing.id)

        if not image and not template_id:
            console.print("[red bold]Error:[/red bold] Provide either image or template_id")
            raise ValueError("Provide either image or template_id")

        # Select GPU
        if gpu:
            gpu_type_id = gpu
            gpu_display = gpu
        else:
            best = self.gpu.find_best(min_vram=min_vram, max_price=max_price, prefer=prefer)
            if not best:
                msg = f"No available GPU with {min_vram}GB+ VRAM"
                if max_price:
                    msg += f" under ${max_price}/hr"
                console.print(f"[red bold]{msg}[/red bold]")
                raise RuntimeError(msg)
            gpu_type_id = best.id
            gpu_display = f"{best.display_name} ({best.memory_gb}GB, ${best.best_price:.2f}/hr)"

            _, explanation = compute_gpu_score(
                best, vram_needed=min_vram,
                all_gpus=self.gpu.filter(min_vram=min_vram, available_only=True, max_price=max_price),
            )
            console.print(f"  [green]Auto-selected:[/green] {gpu_display}")
            console.print(f"  [dim]Reason: {explanation}[/dim]")

        # Show creation details
        details = f"[bold]GPU:[/bold]              {gpu_display}\n"
        details += f"[bold]Image:[/bold]            {image or 'template'}\n"
        details += f"[bold]Container disk:[/bold]   {container_disk_gb}GB"
        if volume_gb > 0:
            details += f"\n[bold]Workspace volume:[/bold] {volume_gb}GB"
        if network_volume_id:
            details += f"\n[bold]Network volume:[/bold]   {network_volume_id}"
        if env:
            details += f"\n[bold]Env vars:[/bold]         {len(env)} set"

        console.print(Panel(details, title=f"Creating pod [cyan bold]{name}[/cyan bold]", border_style="blue"))

        pod = self.pod.create(
            name=name,
            gpu_type_id=gpu_type_id,
            image=image or "",
            gpu_count=gpu_count,
            disk_gb=container_disk_gb,
            volume_gb=volume_gb,
            ports=ports,
            network_volume_id=network_volume_id,
            env=env,
            cloud_type=cloud_type,
            docker_args=docker_args,
            template_id=template_id,
        )
        console.print(f"  [green]Pod created:[/green] [bold]{pod.id}[/bold]")

        if wait:
            pod = self.pod.wait_for_running(pod.id, timeout=timeout, on_timeout="return")
            if pod.is_running:
                console.print(f"  [green bold]Pod ready![/green bold] {pod.id}")
            else:
                console.print(f"  [yellow]Pod not yet ready (status: {pod.status}). It may still be starting.[/yellow]")

        return pod

    def stop(self, pod_id: str = None, name: str = None) -> PodInfo:
        pod_id = self._resolve_pod_id(pod_id, name)
        console.print(f"[yellow]Stopping pod {pod_id}...[/yellow]")
        result = self.pod.stop(pod_id)
        console.print(f"[green]Stopped.[/green] Data preserved on volume.")
        return result

    def resume(self, pod_id: str = None, name: str = None, gpu_count: int = 1, wait: bool = True, timeout: int = 900) -> PodInfo:
        pod_id = self._resolve_pod_id(pod_id, name)
        console.print(f"[cyan]Resuming pod {pod_id}...[/cyan]")
        self.pod.resume(pod_id, gpu_count=gpu_count)
        if wait:
            pod = self.pod.wait_for_running(pod_id, timeout=timeout, on_timeout="return")
            if pod.is_running:
                console.print(f"[green bold]Pod ready![/green bold]")
            return pod
        return self.pod.get(pod_id)

    def terminate(self, pod_id: str = None, name: str = None):
        pod_id = self._resolve_pod_id(pod_id, name)
        console.print(f"[red]Terminating pod {pod_id}...[/red]")
        self.pod.terminate(pod_id)
        console.print("[green]Terminated.[/green]")

    def stop_all(self):
        pods = self.pod.get_all()
        running = [p for p in pods if p.is_running]
        if not running:
            console.print("[dim]No running pods.[/dim]")
            return
        for pod in running:
            console.print(f"  [yellow]Stopping {pod.name} ({pod.id})...[/yellow]")
            self.pod.stop(pod.id)
        console.print(f"[green]Stopped {len(running)} pods.[/green]")

    def cleanup(self):
        pods = self.pod.get_all()
        stopped = [p for p in pods if not p.is_running]
        if not stopped:
            console.print("[dim]No stopped pods to clean up.[/dim]")
            return
        for pod in stopped:
            console.print(f"  [red]Terminating {pod.name} ({pod.id})...[/red]")
            self.pod.terminate(pod.id)
        console.print(f"[green]Cleaned up {len(stopped)} pods.[/green]")

    def terminate_all(self):
        pods = self.pod.get_all()
        if not pods:
            console.print("[dim]No pods.[/dim]")
            return
        for pod in pods:
            console.print(f"  [red]Terminating {pod.name} ({pod.id})...[/red]")
            self.pod.terminate(pod.id)
        console.print(f"[green]Terminated {len(pods)} pods.[/green]")

    # ─── Account & Billing ────────────────────────────────────────────────

    def balance(self, show: bool = True) -> dict:
        from runpod.api.graphql import run_graphql_query
        query = 'query { myself { currentSpendPerHr clientBalance hostBalance minBalance underBalance referralEarned } }'
        result = run_graphql_query(query)
        data = result.get("data", {}).get("myself", {})

        info = {
            "credit_balance": round(data.get("clientBalance", 0), 2),
            "current_spend_per_hr": round(data.get("currentSpendPerHr", 0), 4),
            "referral_earned": round(data.get("referralEarned", 0), 2),
            "under_balance": data.get("underBalance", False),
        }

        if info["current_spend_per_hr"] > 0:
            info["hours_remaining"] = round(info["credit_balance"] / info["current_spend_per_hr"], 1)
        else:
            info["hours_remaining"] = None

        if show:
            content = Text()
            content.append(f"Credits:       ${info['credit_balance']:.2f}\n", style="bold")
            content.append(f"Spend rate:    ${info['current_spend_per_hr']:.4f}/hr\n")
            if info["hours_remaining"]:
                content.append(f"Remaining:     {info['hours_remaining']}h\n", style="cyan")
            content.append(f"Referrals:     ${info['referral_earned']:.2f}")

            border = "red" if info["under_balance"] else "green"
            title = "Account Balance"
            if info["under_balance"]:
                title += " [red bold]⚠ LOW[/red bold]"
            console.print(Panel(content, title=title, border_style=border))

        return info

    # ─── Status ───────────────────────────────────────────────────────────

    def status(self, show: bool = True) -> dict:
        billing = self.balance(show=False)
        pods = self.pod.get_all()
        running = [p for p in pods if p.is_running]
        stopped = [p for p in pods if not p.is_running]
        total_cost = sum(p.cost_per_hour for p in running)

        summary = {
            "credit_balance": billing["credit_balance"],
            "current_spend_per_hr": billing["current_spend_per_hr"],
            "hours_remaining": billing["hours_remaining"],
            "total_pods": len(pods),
            "running": len(running),
            "stopped": len(stopped),
            "running_cost_per_hour": round(total_cost, 2),
            "pods": [
                {"id": p.id, "name": p.name, "status": p.status, "gpu": p.gpu_type, "cost_per_hour": p.cost_per_hour}
                for p in pods
            ],
        }

        if show:
            content = Text()
            content.append(f"Balance: ${summary['credit_balance']:.2f}", style="bold")
            content.append(f"   Spend: ${summary['current_spend_per_hr']:.4f}/hr\n")
            if summary["hours_remaining"]:
                content.append(f"Hours left: {summary['hours_remaining']}h\n", style="cyan")
            content.append(f"Pods: ", style="bold")
            content.append(f"{summary['running']} running", style="green")
            content.append(f", {summary['stopped']} stopped", style="dim")

            if running:
                content.append("\n")
                for p in running:
                    content.append(f"\n  ● {p.name}", style="green")
                    content.append(f"  {p.gpu_type}  ${p.cost_per_hour:.2f}/hr", style="dim")

            console.print(Panel(content, title="RunPod Status", border_style="blue"))

        return summary

    # ─── Internal ─────────────────────────────────────────────────────────

    def _resolve_pod_id(self, pod_id: str = None, name: str = None) -> str:
        if pod_id:
            return pod_id
        if name:
            pod = self.pod.find_by_name(name)
            if not pod:
                console.print(f"[red]No pod found with name '{name}'[/red]")
                raise ValueError(f"No pod found with name '{name}'")
            return pod.id
        console.print("[red]Provide either pod_id or name[/red]")
        raise ValueError("Provide either pod_id or name")
