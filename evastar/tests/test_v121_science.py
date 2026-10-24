"""T63 - v1.21: protein/chem/RL/hardware plugins."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness.harness_core import Context
from harness.harness_plugins import (HardwarePlugin, PresetPlugin,
                                     ProteinPlugin, RLPlugin)


def test_protein_ops():
    ctx = Context()
    ctx.use(PresetPlugin("full", model_fn=None))
    ctx.use(ProteinPlugin())
    assert ctx.protein.validate("ACDEFGHIK")["ok"]
    mw = ctx.protein.mol_weight("ACDEFGHIK")
    assert 900 < mw < 1100
    h = ctx.protein.hydrophobicity("ILV")
    assert h > 3.0
    fold = ctx.protein.fold_toy("ILVAAAAAADEF")
    assert fold["n_core"] >= 3
    print("PASS test_protein_ops")


def test_chem_ops():
    ctx = Context()
    ctx.use(ProteinPlugin())
    assert ctx.chem.formula_weight("H2O") == 18.015
    assert abs(ctx.chem.formula_weight("C6H12O6") - 180.156) < 0.01
    ph = ctx.chem.ph(buffer_conc=0.1, acid_conc=0.01, pka=4.76)
    assert ph > 5.0 and ph < 6.0
    print("PASS test_chem_ops")


def test_rl_reward():
    ctx = Context()
    ctx.use(RLPlugin())
    adj = ctx.rl.rlcd_reward([1.0, 0.0], [0.9, 0.2], [1.0, 0.0], lam=2.0)
    assert adj[0] < 1.0 and adj[1] < 0.0
    print("PASS test_rl_reward")


def test_hardware_stubs():
    ctx = Context()
    ctx.use(HardwarePlugin())
    r = ctx.hw.arduino_write("/dev/ttyUSB0", "LED ON")
    assert "error" in r or "ok" in r
    r2 = ctx.hw.gpio_write(18, 1)
    assert "error" in r2
    print("PASS test_hardware_stubs")


if __name__ == "__main__":
    test_protein_ops()
    test_chem_ops()
    test_rl_reward()
    test_hardware_stubs()
    print("\nv1.21 tests done")
