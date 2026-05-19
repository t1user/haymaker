from dataclasses import dataclass


@dataclass(frozen=True)
class SyncResult:
    """Outcome of broker/local synchronization."""

    ok: bool
    reason: str = "ok"
