import sys, torch
sys.path.insert(0, "/mnt/agents/output")
# GPUMonitor double instantiation with prometheus installed?
try:
    import prometheus_client
    from helioslm_v5.src.inference.vllm_engine import GPUMonitor
    m1 = GPUMonitor(export_prometheus=True)
    try:
        m2 = GPUMonitor(export_prometheus=True)
        print("second GPUMonitor: OK")
    except Exception as e:
        print("second GPUMonitor raises:", type(e).__name__, str(e)[:80])
except ImportError:
    print("prometheus_client not installed; skipping")
