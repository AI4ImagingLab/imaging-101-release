"""Docker container sandbox for executing agent-generated code.

The original task directory is mounted **read-only** at ``/workspace_src``.
Selected paths are copied into the writable ``/workspace`` where the agent
operates.  When the container is removed all agent-written files vanish —
the original task directory is never modified.
"""

from __future__ import annotations

import logging
import subprocess
import uuid
from pathlib import Path

log = logging.getLogger(__name__)

MAX_OUTPUT_CHARS = 5000


class DockerRunner:
    """Manages a persistent Docker container for sandboxed execution.

    Workflow: ``start()`` → repeated ``exec()`` / ``write_file()`` → ``stop()``.
    """

    def __init__(self, image: str, task_dir: Path, timeout: int = 600) -> None:
        self.image = image
        self.task_dir = task_dir.resolve()
        self.timeout = timeout
        self.container: str | None = None

    # ------------------------------------------------------------------
    def start(self, visible_paths: list[str] | None = None) -> None:
        """Create the sandbox container and seed the workspace.

        Parameters
        ----------
        visible_paths
            Relative paths (files or dirs) inside the task directory that
            the agent is allowed to see.  If *None*, the entire task
            directory is copied.
        """
        name = f"imaging101-{uuid.uuid4().hex[:8]}"
        cmd = [
            "docker", "run", "-d",
            "--name", name,
            "-v", f"{self.task_dir}:/workspace_src:ro",
            "-w", "/workspace",
            "--memory=4g",
            "--cpus=2",
            self.image,
            "sleep", "infinity",
        ]
        log.info("Starting container %s", name)
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        self.container = name

        # Seed workspace with visible files
        if visible_paths:
            for p in visible_paths:
                # cp -a preserves structure; || true in case path doesn't exist
                self.exec(f"cp -a /workspace_src/{p} /workspace/{p} 2>/dev/null || true")
        else:
            self.exec("cp -a /workspace_src/. /workspace/")

        # Install per-task requirements if present
        req = self.task_dir / "requirements.txt"
        if req.exists():
            log.info("Installing task requirements…")
            self.exec("pip install -q -r /workspace_src/requirements.txt")

    # ------------------------------------------------------------------
    def exec(self, command: str) -> tuple[str, int]:
        """Run *command* inside the container.

        Returns (stdout+stderr, exit_code).  Output is truncated to
        ``MAX_OUTPUT_CHARS`` characters.
        """
        assert self.container, "Container not started"
        cmd = [
            "docker", "exec", self.container,
            "bash", "-c", f"timeout {self.timeout} {command}",
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=self.timeout + 30,
        )
        output = (result.stdout + result.stderr).strip()
        if len(output) > MAX_OUTPUT_CHARS:
            output = output[:MAX_OUTPUT_CHARS] + "\n… [truncated]"
        return output, result.returncode

    # ------------------------------------------------------------------
    def write_file(self, path: str, content: str) -> None:
        """Write *content* to *path* inside the container."""
        assert self.container, "Container not started"
        # Ensure parent directory exists
        parent = str(Path(path).parent)
        if parent and parent != ".":
            self.exec(f"mkdir -p {parent}")
        # Use docker exec -i + stdin pipe to avoid shell escaping issues
        proc = subprocess.run(
            ["docker", "exec", "-i", self.container, "bash", "-c", f"cat > {path}"],
            input=content, text=True, capture_output=True,
        )
        if proc.returncode != 0:
            log.error("write_file failed: %s", proc.stderr)

    # ------------------------------------------------------------------
    def read_file(self, path: str) -> str:
        """Read a file from the container."""
        output, rc = self.exec(f"cat {path}")
        if rc != 0:
            return f"[Error reading {path}]: {output}"
        return output

    # ------------------------------------------------------------------
    def stop(self) -> None:
        """Remove the container (and all agent-written files)."""
        if self.container:
            log.info("Stopping container %s", self.container)
            subprocess.run(
                ["docker", "rm", "-f", self.container],
                capture_output=True, text=True,
            )
            self.container = None
