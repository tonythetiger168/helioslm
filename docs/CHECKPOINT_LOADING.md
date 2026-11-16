# Pilot checkpoint loading boundary

Tracks issue #273. Pilot training resume and `init_from`, and
`examples/package_tiers.py` now explicitly use `torch.load(weights_only=True)`.
The pilot's existing dictionary of model tensors, AdamW state, step integer
and config string remains the format. A raw model state dictionary remains
supported for `init_from`. Unsupported pickle globals are rejected; there is
no retry with unrestricted pickle and no extra global allowlist.

This is one bounded deserialization fix, not the completion of issue #273.
Restricted loading does not authenticate the model source, prove that weights
are benign, limit tensor allocation size, or replace a sandbox. Use only
checkpoints from a trusted source. A trustworthy input manifest/revision,
pre-load hash/size verification, remote-code opt-in gates and a reviewed
legacy conversion workflow remain separate work. An output package checksum
does not establish input provenance. Never turn off restricted loading merely
to make an unknown checkpoint load.

CPU regression commands (use the pinned CI and test-extra dependencies):

```sh
python -m pytest helioslm_v5/tests/test_pretrain_pilot.py helioslm_v5/tests/test_package_tiers.py -q
python -m pytest helioslm_v5/tests/ -q -n 4 --ignore=helioslm_v5/tests/test_v5.py
python integration_test_v51.py
```

The new pilot regression loads a genuine tiny training checkpoint into the
package loader, exercises both init formats and resume, then checks all three
entry points reject a synthetic non-allowlisted global without executing its
harmless marker function or rewriting the existing checkpoint/log. Existing
bitwise resume tests remain in place. The regression is also registered in the
daily gate. No external weights, API, GPU or paid resources are needed.
