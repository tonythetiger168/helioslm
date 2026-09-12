"""Audit logging for compliance and security."""
import json
import time
import hashlib
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass, asdict


@dataclass
class AuditRecord:
    """Single audit log record."""
    timestamp: float
    request_id: str
    client_id: str
    endpoint: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    moderation_result: Optional[Dict]
    error: Optional[str]
    latency_ms: float
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


class AuditLogger:
    """
    Compliance-grade audit logging.

    Features:
      - Structured JSON logging
      - Automatic rotation
      - Tamper-evident hashing (optional)
      - PII redaction
    """

    def __init__(
        self,
        log_dir: str = "logs/audit",
        redact_pii: bool = True,
        enable_hash_chain: bool = False,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.redact_pii = redact_pii
        self.enable_hash_chain = enable_hash_chain

        self.current_log = self.log_dir / f"audit_{time.strftime('%Y%m%d')}.jsonl"
        self.last_hash = "0" * 64

    def log(self, record: AuditRecord):
        """Log an audit record."""
        data = asdict(record)

        # Redact PII if enabled
        if self.redact_pii:
            data = self._redact_pii(data)

        # Add hash chain for tamper evidence
        if self.enable_hash_chain:
            record_str = json.dumps(data, sort_keys=True)
            current_hash = hashlib.sha256(f"{self.last_hash}{record_str}".encode()).hexdigest()
            data["_hash"] = current_hash
            data["_prev_hash"] = self.last_hash
            self.last_hash = current_hash

        # Write to log
        with open(self.current_log, "a") as f:
            f.write(json.dumps(data, default=str) + "\n")

    def _redact_pii(self, data: Dict) -> Dict:
        """Redact potential PII from log data."""
        # Redact IP addresses
        if "ip_address" in data and data["ip_address"]:
            data["ip_address"] = self._hash_value(data["ip_address"])

        # Redact user agent (may contain identifiable info)
        if "user_agent" in data and data["user_agent"]:
            data["user_agent"] = data["user_agent"][:50] + "..."

        return data

    def _hash_value(self, value: str) -> str:
        """Hash a value for pseudonymization."""
        return hashlib.sha256(value.encode()).hexdigest()[:16]

    def query(self, start_time: float, end_time: float, client_id: Optional[str] = None) -> list:
        """Query audit logs by time range."""
        results = []

        for log_file in self.log_dir.glob("audit_*.jsonl"):
            with open(log_file) as f:
                for line in f:
                    record = json.loads(line)
                    ts = record.get("timestamp", 0)
                    if start_time <= ts <= end_time:
                        if client_id is None or record.get("client_id") == client_id:
                            results.append(record)

        return results
