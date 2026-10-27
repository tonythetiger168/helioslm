"""hsh H13: network egress audit for tool subprocesses (``ctx.egress``).

A local model that runs commands should not reach the network unless the
operator said so, and when it tries, the operator should see where. With
the ``Egress`` service mounted, ``bash`` and ``run_code`` children get
two layers:

1. a proxy on 127.0.0.1 named in ``HTTP_PROXY`` / ``HTTPS_PROXY`` /
   ``ALL_PROXY`` (and the lowercase forms). curl, pip, git over HTTPS,
   requests and urllib go through it. ``CONNECT`` to an allowed host is
   tunnelled and plain HTTP to an allowed host is forwarded; any other
   host gets 403 and a record;
2. a Python audit hook on ``socket.connect`` and ``socket.getaddrinfo``,
   installed by a ``sitecustomize`` on ``PYTHONPATH`` for Python started
   from ``bash`` and prepended to the ``run_code`` child. A connection to
   a host that is not loopback and not allowed is recorded and, in
   ``deny`` mode, refused with ``PermissionError``.

Config: ``mode`` ``"deny"`` (default) or ``"audit"`` (record, let
through); ``allow``: host patterns (``fnmatch``, e.g.
``"*.huggingface.co"``); ``isolate``: run ``bash`` under ``unshare -rn``
(Linux user namespaces) so the child has loopback only and the allow
list cannot apply.

Without ``isolate`` this is an audit trail, not a sandbox: a program that
ignores the proxy variables and is not Python can still connect
directly. Every record is ``{"layer", "host", "port", "allowed"}``;
``drain()`` returns the records since the last drain (the tools attach
them to their result), ``report()`` sums all of them per host.
"""
from __future__ import annotations

import fnmatch
import json
import select
import shutil
import socket
import tempfile
import threading
from pathlib import Path

from .kernel import Service

LOOPBACK = ("localhost", "127.0.0.1", "::1")

# shared by sitecustomize.py and the run_code child; reads its settings
# from the environment so the child needs nothing else
HOOK = r'''
def _hsh_egress():
    import fnmatch, json, os, sys
    log = os.environ.get("HSH_EGRESS_LOG")
    if not log:
        return
    allow = json.loads(os.environ.get("HSH_EGRESS_ALLOW", "[]"))
    deny = os.environ.get("HSH_EGRESS_MODE", "deny") == "deny"
    proxy = int(os.environ.get("HSH_EGRESS_PROXY_PORT", "0"))
    loop = ("localhost", "127.0.0.1", "::1", "", None)

    ips = set()          # addresses of allowed names, resolved in the hook
    busy = []

    def check(host, port, resolve=False):
        host = host.decode() if isinstance(host, bytes) else host
        if host in loop or host in ips or (isinstance(host, str) and host.startswith("127.")):
            return
        ok = any(fnmatch.fnmatch(str(host).lower(), p) for p in allow)
        with open(log, "a") as fh:
            fh.write(json.dumps({"layer": "python", "host": str(host),
                                 "port": port, "allowed": ok}) + "\n")
        if ok and resolve and not busy:
            import socket
            busy.append(1)
            try:
                ips.update(a[4][0] for a in socket.getaddrinfo(host, port))
            except OSError:
                pass
            finally:
                busy.pop()
        if deny and not ok:
            raise PermissionError(f"network access to {host} blocked by hsh egress")

    def hook(event, args):
        if busy:
            return
        if event == "socket.getaddrinfo":
            check(args[0], args[1], resolve=True)
        elif event == "socket.connect":
            addr = args[1]
            if isinstance(addr, tuple) and len(addr) >= 2 and addr[1] != proxy:
                check(addr[0], addr[1])
    sys.addaudithook(hook)
_hsh_egress()
del _hsh_egress
'''


class _Proxy(threading.Thread):
    def __init__(self, egress):
        super().__init__(daemon=True)
        self.egress = egress
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(16)
        self.port = self.sock.getsockname()[1]

    def run(self):
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            threading.Thread(target=self.handle, args=(conn,), daemon=True).start()

    def handle(self, conn):
        try:
            head = b""
            while b"\r\n\r\n" not in head and len(head) < 65536:
                chunk = conn.recv(4096)
                if not chunk:
                    return
                head += chunk
            line = head.split(b"\r\n", 1)[0].decode("latin-1")
            method, target, _ = (line.split(" ") + ["", ""])[:3]
            if method == "CONNECT":
                host, _, port = target.rpartition(":")
                port = int(port or 443)
            else:
                rest = target.split("://", 1)[-1]
                hostport, _, path = rest.partition("/")
                host, _, port = hostport.partition(":")
                port = int(port or 80)
            host = host.strip("[]")
            if not self.egress.record("proxy", host, port):
                conn.sendall(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\n"
                             b"Connection: close\r\n\r\n")
                return
            up = socket.create_connection((host, port), timeout=30)
            if method == "CONNECT":
                conn.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
                rest = head.split(b"\r\n\r\n", 1)[1]
            else:
                rest = head.replace(target.encode("latin-1"),
                                    ("/" + path).encode("latin-1"), 1)
            if rest:
                up.sendall(rest)
            _relay(conn, up)
        except (OSError, ValueError):
            pass
        finally:
            conn.close()

    def close(self):
        self.sock.close()


def _relay(a, b):
    socks = [a, b]
    try:
        while True:
            ready, _, _ = select.select(socks, [], [], 60)
            if not ready:
                return
            for s in ready:
                data = s.recv(65536)
                if not data:
                    return
                (b if s is a else a).sendall(data)
    finally:
        b.close()


class Egress(Service):
    key = "egress"

    def __init__(self, ctx, config=None):
        super().__init__(ctx, config)
        cfg = config or {}
        self.mode = cfg.get("mode", "deny")
        if self.mode not in ("deny", "audit"):
            raise ValueError(f"egress mode must be deny or audit, not {self.mode!r}")
        self.allow = [p.lower() for p in cfg.get("allow", [])]
        self.isolate = bool(cfg.get("isolate")) and shutil.which("unshare") is not None
        self.records = []
        self._seen = 0
        self._lock = threading.Lock()
        self._dir = None
        self._proxy = None

    def start(self):
        self._dir = Path(tempfile.mkdtemp(prefix="hsh-egress-"))
        (self._dir / "sitecustomize.py").write_text(HOOK)
        self.log = self._dir / "python.jsonl"
        self.log.touch()
        self._read = 0
        self._proxy = _Proxy(self)
        self._proxy.start()

    def stop(self):
        if self._proxy:
            self._proxy.close()
        if self._dir:
            shutil.rmtree(self._dir, ignore_errors=True)

    def allowed(self, host):
        host = (host or "").lower()
        if host in LOOPBACK or host.startswith("127."):
            return True
        return any(fnmatch.fnmatch(host, p) for p in self.allow)

    def record(self, layer, host, port):
        """Note one attempt; return whether it may proceed."""
        ok = self.allowed(host)
        with self._lock:
            self.records.append({"layer": layer, "host": host, "port": port,
                                 "allowed": ok})
        return ok or self.mode == "audit"

    def env(self):
        """Variables for a tool child (merge into its scrubbed env)."""
        url = f"http://127.0.0.1:{self._proxy.port}"
        env = {"HSH_EGRESS_LOG": str(self.log), "HSH_EGRESS_MODE": self.mode,
               "HSH_EGRESS_ALLOW": json.dumps(self.allow),
               "HSH_EGRESS_PROXY_PORT": str(self._proxy.port),
               "PYTHONPATH": str(self._dir), "NO_PROXY": "localhost,127.0.0.1",
               "no_proxy": "localhost,127.0.0.1"}
        for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
            env[k] = env[k.lower()] = url
        return env

    def wrap(self, argv):
        """``argv`` under ``unshare -rn`` when ``isolate`` is on."""
        return ["unshare", "-rn", *argv] if self.isolate else argv

    def _collect(self):
        with open(self.log) as fh:
            fh.seek(self._read)
            lines = fh.readlines()
            self._read = fh.tell()
        with self._lock:
            for ln in lines:
                if ln.strip():
                    self.records.append(json.loads(ln))

    def drain(self):
        """Records since the previous drain."""
        self._collect()
        with self._lock:
            new = self.records[self._seen:]
            self._seen = len(self.records)
        return new

    def report(self):
        """``{host: {"allowed": n, "blocked": n}}`` over the whole session."""
        self._collect()
        out = {}
        for r in self.records:
            row = out.setdefault(r["host"], {"allowed": 0, "blocked": 0})
            row["allowed" if r["allowed"] else "blocked"] += 1
        return out


def egress_note(records, mode="deny"):
    """One line for a tool's rendered output, or ''."""
    hosts = sorted({f"{r['host']}:{r['port']}" for r in records if not r["allowed"]})
    if not hosts:
        return ""
    verb = "blocked" if mode == "deny" else "recorded (not on the allow list)"
    return f"[network {verb} by the harness: " + ", ".join(hosts) + "]"
