"""Verify the Open IP APB wrapper with the Open VIP APB test components.

Inputs are caller-supplied checkouts, the extracted APB-VIP directory, and
the combined UVM library used by HeliosLM's UVM benchmark. No downloads.
"""

import argparse
import json
import re
import subprocess
from pathlib import Path


HERE = Path(__file__).resolve().parent
DIRECT_PASS = "APB_PORTFOLIO_DIRECT_PASS checks=10"
UVM_PASS = "APB_PORTFOLIO_UVM_PASS checks=7"


def run_logged(command, out, name, timeout):
    log = out / f"{name}.log"
    with log.open("wb") as handle:
        result = subprocess.run(command, cwd=out, stdout=handle,
                                stderr=subprocess.STDOUT, timeout=timeout)
    return result.returncode, log.read_text(encoding="utf-8", errors="replace")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def direct(rtl, out, args):
    exe = out / "direct.vvp"
    cmd = [args.iverilog, "-g2012", "-s", "apb_portfolio_direct_tb", "-o", exe,
           *rtl, HERE / "bridge.sv", HERE / "direct_tb.sv"]
    code, _ = run_logged(cmd, out, "direct_compile", 60)
    require(code == 0, "Icarus compile failed. See direct_compile.log.")
    code, log = run_logged([args.vvp, exe], out, "direct_sim", 30)
    require(code == 0 and DIRECT_PASS in log, "APB checks failed. See direct_sim.log.")


def uvm(rtl, out, args):
    vip = args.vip.resolve(strict=True)
    uvm_source = args.uvm.resolve(strict=True)
    include_dirs = ["rtl", "vip/ahb", "vip/apb", "vip/axi", "vip/base"]
    (out / "uvm_macros.svh").write_text(
        "// Macros come from the combined UVM source compiled first.\n", encoding="utf-8")
    sources = [uvm_source, vip / "rtl/amba_vip_if.sv",
               vip / "vip/base/amba_vip_pkg.sv", *rtl, HERE / "bridge.sv", HERE / "uvm_tb.sv"]
    cmd = [args.verilator, "--binary", "--timing", "--assert", "-Wno-fatal",
           "--timescale", "1ns/1ps", "--top-module", "apb_portfolio_uvm_tb",
           "-j", str(args.jobs), "-CFLAGS", "-O0", f"-I{out}",
           *(f"-I{vip / path}" for path in include_dirs), *sources]
    code, _ = run_logged(cmd, out, "uvm_compile", args.build_timeout)
    require(code == 0, "UVM compile failed. See uvm_compile.log.")
    exe = out / "obj_dir/Vapb_portfolio_uvm_tb"
    code, log = run_logged([exe], out, "uvm_sim", 60)
    reports_clean = all(re.search(rf"(?m)^\s*UVM_{severity}\s*:\s*0\s*$", log)
                        for severity in ("WARNING", "ERROR", "FATAL"))
    require(code == 0 and UVM_PASS in log and reports_clean,
            "UVM run failed or reported diagnostics. See uvm_sim.log.")
    _, log = run_logged([exe, "+INJECT_BAD_ID"], out, "uvm_negative", 60)
    require("[APB_DATA]" in log and "unexpected response at 00000000" in log
            and UVM_PASS not in log, "UVM failed to reject the corrupted ID response.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--open-ip", type=Path, required=True)
    parser.add_argument("--vip", type=Path)
    parser.add_argument("--uvm", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iverilog", default="iverilog")
    parser.add_argument("--vvp", default="vvp")
    parser.add_argument("--verilator", default="verilator")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--build-timeout", type=int, default=900)
    args = parser.parse_args()
    if bool(args.vip) != bool(args.uvm):
        parser.error("provide both --vip and --uvm to run the UVM check")
    if args.jobs < 1 or args.build_timeout < 1:
        parser.error("--jobs and --build-timeout must be positive")
    rtl = [(args.open_ip / f"rtl/{name}.sv").resolve(strict=True)
           for name in ("APB_top", "APB_apb")]
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    receipt = {"direct": "pending", "uvm": "pending" if args.uvm else "not_run"}
    stage = "direct"
    try:
        direct(rtl, out, args)
        receipt[stage] = "passed"
        if args.uvm:
            stage = "uvm"
            uvm(rtl, out, args)
            receipt[stage] = "passed"
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        receipt[stage] = "failed"
        receipt["error"] = str(error)
        raise
    finally:
        (out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
