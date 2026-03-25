"""
InferAll Dashboard
-----------------------
Textual TUI for monitoring and managing the InferAll server.

Features:
- Real-time GPU monitoring (VRAM, temp, utilization)
- Per-model request queue stats
- Live request log
- Model management (load/unload/pull/delete)
- API key management
- Server performance metrics
"""

import json
import time
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.reactive import reactive
from textual.timer import Timer
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Input,
    Label,
    ProgressBar,
    RichLog,
    Static,
    TabbedContent,
    TabPane,
)


class ServerClient:
    """HTTP client for the InferAll API."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip("/")

    def get(self, path: str) -> dict:
        url = f"{self.base_url}{path}"
        req = Request(url, method="GET")
        with urlopen(req, timeout=5) as resp:
            return json.loads(resp.read())

    def post(self, path: str, data: dict = None) -> dict:
        url = f"{self.base_url}{path}"
        body = json.dumps(data or {}).encode()
        req = Request(url, data=body, headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(req, timeout=30) as resp:
            return json.loads(resp.read())

    def delete(self, path: str) -> dict:
        url = f"{self.base_url}{path}"
        req = Request(url, method="DELETE")
        with urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())

    def is_alive(self) -> bool:
        try:
            self.get("/health")
            return True
        except Exception:
            return False


# =============================================================================
# Widgets
# =============================================================================

class GPUPanel(Static):
    """Real-time GPU status display."""

    def render(self) -> str:
        return self.app._gpu_text

class QueuePanel(Static):
    """Per-model request queue stats."""

    def render(self) -> str:
        return self.app._queue_text

class PerformancePanel(Static):
    """Server performance metrics."""

    def render(self) -> str:
        return self.app._perf_text

class ServerStatusBar(Static):
    """Server connection status."""

    def render(self) -> str:
        return self.app._status_text


# =============================================================================
# Main App
# =============================================================================

class DashboardApp(App):
    """InferAll Dashboard TUI."""

    TITLE = "InferAll Dashboard"
    CSS = """
    Screen {
        layout: vertical;
    }
    #top-bar {
        height: 1;
        background: $accent;
        color: $text;
        text-align: center;
    }
    #main-content {
        height: 1fr;
    }
    #gpu-panel {
        width: 35;
        border: solid $accent;
        padding: 0 1;
    }
    #right-column {
        width: 1fr;
    }
    #queue-panel {
        height: 40%;
        border: solid $accent;
        padding: 0 1;
    }
    #perf-panel {
        height: 20%;
        border: solid $accent;
        padding: 0 1;
    }
    #log-panel {
        height: 40%;
        border: solid $accent;
    }
    #models-table {
        height: 1fr;
    }
    .tab-pane {
        padding: 1;
    }
    #model-input {
        margin: 1 0;
    }
    Footer {
        background: $panel;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("1", "tab_dashboard", "Dashboard"),
        Binding("2", "tab_models", "Models"),
        Binding("3", "tab_keys", "Keys"),
    ]

    # Reactive state
    _gpu_text = reactive("[b]GPU Status[/b]\nConnecting...")
    _queue_text = reactive("[b]Request Queues[/b]\nConnecting...")
    _perf_text = reactive("[b]Performance[/b]\nConnecting...")
    _status_text = reactive("Connecting...")

    def __init__(self, server_url: str = "http://127.0.0.1:8000"):
        super().__init__()
        self.client = ServerClient(server_url)
        self._total_requests = 0
        self._total_errors = 0
        self._start_time = time.time()

    def compose(self) -> ComposeResult:
        yield Header()
        with TabbedContent():
            with TabPane("Dashboard", id="tab-dashboard"):
                yield ServerStatusBar(id="top-bar")
                with Horizontal(id="main-content"):
                    yield GPUPanel(id="gpu-panel")
                    with Vertical(id="right-column"):
                        yield QueuePanel(id="queue-panel")
                        yield PerformancePanel(id="perf-panel")
                        yield RichLog(id="log-panel", highlight=True, markup=True, wrap=True)
            with TabPane("Models", id="tab-models"):
                yield Label("[b]Pulled Models[/b]  (R=Refresh)")
                yield DataTable(id="models-table")
                yield Input(placeholder="Enter model name to pull (e.g., llama3.1 or meta-llama/Llama-3-8B)", id="model-input")
                yield Label(id="model-action-status")
            with TabPane("Keys", id="tab-keys"):
                yield Label(
                    "[b]API Keys[/b]  (R=Refresh)\n\n"
                    "API keys control who can access your InferAll server.\n"
                    "Each key has its own rate limits (requests per minute/day) and priority level.\n"
                    "High-priority keys get served first when the server is under load.\n\n"
                    "[dim]Priority levels: 0=Normal  1=High (skip queue)  2=Critical (always served first)\n"
                    "Clients authenticate via: Authorization: Bearer <key>\n"
                    "Keys are shown only once at creation — store them securely.[/dim]\n"
                )
                yield DataTable(id="keys-table")
                yield Input(placeholder="Enter a name for the new key (e.g., 'frontend-app', 'john-dev', 'production')", id="key-input")
                yield Label(id="key-action-status")
        yield Footer()

    def on_mount(self) -> None:
        """Start periodic refresh."""
        self.set_interval(2.0, self._update_dashboard)
        self.set_interval(10.0, self._update_models_table)
        self.set_interval(10.0, self._update_keys_table)
        # Initial load
        self.call_later(self._update_dashboard)
        self.call_later(self._update_models_table)
        self.call_later(self._update_keys_table)

    # -------------------------------------------------------------------------
    # Dashboard Updates
    # -------------------------------------------------------------------------

    def _update_dashboard(self) -> None:
        """Refresh dashboard panels from server API."""
        log = self.query_one("#log-panel", RichLog)

        try:
            # Health check
            health = self.client.get("/health")
            loaded = health.get("loaded_models", 0)
            caps = sum(1 for v in health.get("capabilities", {}).values() if v)
            self._status_text = (
                f" SERVER: [green]ONLINE[/green]  |  "
                f"Models loaded: {loaded}  |  "
                f"Capabilities: {caps}  |  "
                f"Uptime: {self._format_uptime()}"
            )
        except Exception:
            self._status_text = " SERVER: [red]OFFLINE[/red]  |  Cannot connect to API"
            self._gpu_text = "[b]GPU Status[/b]\n[dim]Server offline[/dim]"
            self._queue_text = "[b]Request Queues[/b]\n[dim]Server offline[/dim]"
            self._perf_text = "[b]Performance[/b]\n[dim]Server offline[/dim]"
            return

        try:
            # Queue stats (includes GPU info)
            stats = self.client.get("/v1/queue/stats")

            # GPU panel
            gpus = stats.get("gpus", [])
            gpu_lines = ["[b]GPU Status[/b]", ""]
            for g in gpus:
                if "error" in g:
                    gpu_lines.append(f"GPU {g['gpu_id']}: [red]Error[/red]")
                    continue
                total = g.get("total_memory_gb", 24)
                used = g.get("used_memory_gb", 0)
                free = g.get("free_memory_gb", 0)
                pct = (used / total * 100) if total > 0 else 0
                bar_len = 20
                filled = int(bar_len * pct / 100)
                bar = "█" * filled + "░" * (bar_len - filled)
                color = "green" if pct < 60 else "yellow" if pct < 85 else "red"

                gpu_lines.append(f"[b]GPU {g['gpu_id']}[/b]: {g.get('name', 'GPU')[:18]}")
                gpu_lines.append(f"  [{color}]{bar}[/{color}] {pct:.0f}%")
                gpu_lines.append(f"  {used:.1f} / {total:.1f} GB")
                temp = g.get("temperature")
                util = g.get("utilization")
                if temp is not None:
                    t_color = "green" if temp < 70 else "yellow" if temp < 85 else "red"
                    gpu_lines.append(f"  Temp: [{t_color}]{temp}°C[/{t_color}]  Util: {util or 0}%")
                gpu_lines.append(f"  Models: {g.get('models_loaded', 0)}")
                gpu_lines.append("")
            self._gpu_text = "\n".join(gpu_lines)

            # Queue panel
            models = stats.get("models", {})
            q_lines = ["[b]Request Queues[/b]", ""]
            total_served = 0
            total_errors = 0
            if not models:
                q_lines.append("[dim]No active queues[/dim]")
            for mid, s in models.items():
                name = mid.split("/")[-1][:25] if "/" in mid else mid[:25]
                active = s.get("active", 0)
                pending = s.get("pending", 0)
                served = s.get("total_served", 0)
                errors = s.get("total_errors", 0)
                avg = s.get("avg_latency_ms", 0)
                total_served += served
                total_errors += errors

                indicator = "[green]●[/green]" if active > 0 else "[dim]○[/dim]"
                q_lines.append(f"  {indicator} {name}")
                q_lines.append(f"    Active:{active} Pending:{pending} Served:{served} Avg:{avg:.0f}ms")
                if errors > 0:
                    q_lines.append(f"    [red]Errors: {errors}[/red]")
            self._queue_text = "\n".join(q_lines)
            self._total_requests = total_served
            self._total_errors = total_errors

            # Performance panel
            uptime = self._format_uptime()
            rps = total_served / max(1, time.time() - self._start_time)
            avg_latencies = [s.get("avg_latency_ms", 0) for s in models.values() if s.get("avg_latency_ms", 0) > 0]
            overall_avg = sum(avg_latencies) / len(avg_latencies) if avg_latencies else 0

            p_lines = [
                "[b]Performance[/b]", "",
                f"  Total requests: {total_served}",
                f"  Total errors:   {total_errors}",
                f"  Avg latency:    {overall_avg:.0f}ms",
                f"  Req/sec:        {rps:.2f}",
            ]
            self._perf_text = "\n".join(p_lines)

        except Exception as e:
            log.write(f"[red]Update error: {e}[/red]")

    def _update_models_table(self) -> None:
        """Refresh the models table."""
        table = self.query_one("#models-table", DataTable)

        try:
            data = self.client.get("/v1/models")
            models = data.get("data", [])
        except Exception:
            return

        table.clear(columns=True)
        table.add_columns("Model", "Task", "Format", "Owner", "Created")

        for m in models:
            model_id = m.get("id", "")
            task = m.get("task", "")
            fmt = m.get("format", "")
            owner = m.get("owned_by", "")
            created = m.get("created", 0)
            date_str = datetime.fromtimestamp(created).strftime("%Y-%m-%d") if created else "-"

            name = model_id.split("/")[-1][:35] if "/" in model_id else model_id[:35]
            table.add_row(name, task, fmt, owner, date_str)

    def _update_keys_table(self) -> None:
        """Refresh the keys table."""
        table = self.query_one("#keys-table", DataTable)

        try:
            from inferall.config import EngineConfig
            from inferall.auth.key_store import KeyStore
            config = EngineConfig.load()
            db_path = config.base_dir / "auth.db"
            if not db_path.exists():
                return
            store = KeyStore(str(db_path))
            keys = store.list_keys()
            store.close()
        except Exception:
            return

        table.clear(columns=True)
        table.add_columns("Prefix", "Name", "Priority", "RPM", "RPD", "Status")

        for k in keys:
            status = "active" if k["enabled"] else "revoked"
            table.add_row(
                k["key_prefix"], k["name"], str(k["priority"]),
                str(k["rate_limit_rpm"]), str(k["rate_limit_rpd"]), status,
            )

    # -------------------------------------------------------------------------
    # Input Handlers
    # -------------------------------------------------------------------------

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle input submissions (model pull, key creation)."""
        if event.input.id == "model-input":
            self._handle_model_pull(event.value)
            event.input.value = ""
        elif event.key_input and event.input.id == "key-input":
            self._handle_key_create(event.value)
            event.input.value = ""

    def _handle_model_pull(self, model_name: str) -> None:
        """Pull a model via CLI."""
        if not model_name.strip():
            return
        status = self.query_one("#model-action-status", Label)
        status.update(f"Pulling {model_name}... (this may take a while)")

        import subprocess
        import shutil
        try:
            # Find inferall in PATH (works with any install method)
            inferall_bin = shutil.which("inferall") or "inferall"
            result = subprocess.run(
                [inferall_bin, "pull", model_name],
                capture_output=True, text=True, timeout=600,
            )
            if result.returncode == 0:
                status.update(f"[green]Pulled {model_name} successfully[/green]")
            else:
                status.update(f"[red]Pull failed: {result.stderr[:100]}[/red]")
        except Exception as e:
            status.update(f"[red]Error: {e}[/red]")

        self.call_later(self._update_models_table)

    def _handle_key_create(self, key_name: str) -> None:
        """Create a new API key."""
        if not key_name.strip():
            return
        status = self.query_one("#key-action-status", Label)
        try:
            from inferall.config import EngineConfig
            from inferall.auth.key_store import KeyStore
            config = EngineConfig.load()
            config.ensure_dirs()
            store = KeyStore(str(config.base_dir / "auth.db"))
            raw_key = store.create_key(name=key_name)
            store.close()
            status.update(f"[green]Key created: {raw_key}[/green]\n[yellow]Save this — shown only once![/yellow]")
        except Exception as e:
            status.update(f"[red]Error: {e}[/red]")
        self.call_later(self._update_keys_table)

    # -------------------------------------------------------------------------
    # Actions
    # -------------------------------------------------------------------------

    def action_refresh(self) -> None:
        """Manual refresh."""
        self._update_dashboard()
        self._update_models_table()
        self._update_keys_table()

    def action_tab_dashboard(self) -> None:
        self.query_one(TabbedContent).active = "tab-dashboard"

    def action_tab_models(self) -> None:
        self.query_one(TabbedContent).active = "tab-models"

    def action_tab_keys(self) -> None:
        self.query_one(TabbedContent).active = "tab-keys"

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _format_uptime(self) -> str:
        elapsed = int(time.time() - self._start_time)
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours > 0:
            return f"{hours}h {minutes}m"
        return f"{minutes}m {seconds}s"


def run_dashboard(server_url: str = "http://127.0.0.1:8000"):
    """Launch the TUI dashboard."""
    app = DashboardApp(server_url=server_url)
    app.run()
