"""Shared request-origin checks for the local harness web UIs (v1 + v2).

Binding to 127.0.0.1 keeps remote hosts out, but any page open in the
user's browser can still reach a loopback server: a DNS-rebinding page
arrives with a foreign Host header, and a cross-site form/fetch POST
carries a foreign Origin (or, from browsers that omit Origin, a
`Sec-Fetch-Site: cross-site`). Stdlib-only, no import side effects.
"""

LOCAL_HOSTS = {"127.0.0.1", "localhost", "[::1]"}


def host_name(value):
    """Hostname part of a Host header / origin authority, lowercased."""
    v = (value or "").strip().lower()
    if v.startswith("["):                      # [::1]:8990
        end = v.find("]")
        return v[:end + 1] if end != -1 else v
    return v.rsplit(":", 1)[0] if ":" in v else v


def request_allowed(headers, state_changing):
    """Host must be loopback on every request. On state-changing ones,
    Origin (when present) must be loopback and Sec-Fetch-Site must not
    be cross-site."""
    if host_name(headers.get("Host")) not in LOCAL_HOSTS:
        return False
    if not state_changing:
        return True
    if (headers.get("Sec-Fetch-Site") or "").strip().lower() == "cross-site":
        return False
    origin = headers.get("Origin")
    if origin is not None:
        scheme, _, authority = origin.partition("://")
        if scheme not in ("http", "https") or \
                host_name(authority) not in LOCAL_HOSTS:
            return False
    return True
