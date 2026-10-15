# Calibration board (2026-10-15)

Schema `calibration-board/1`, tier `measured`, ranked by brier (ascending). `overconf. wrong` = wrong answers with conf ≥ 0.9.

| # | model / artifact | conf type | scored | acc | AUROC | Brier | ECE | conf on wrong | overconf. wrong |
|---|---|---|---|---|---|---|---|---|---|
| 1 | mid_agent_eval_v5 (`mid_agent_eval_v5.json`) | token_prob | 12/12 | 1.0000 | — | 0.0000 | 0.0001 | — | 0/0 |
| 2 | gpt-5.6-luna (`api_probe_luna_2026-10-04.json`) | self_consistency | 12/12 | 0.4167 | 0.8429 | 0.1574 | 0.1389 | 0.3810 | 0/7 |
| 3 | mid_agent_eval_v5_trilogy (`mid_agent_eval_v5_trilogy.json`) | token_prob | 12/12 | 0.7500 | 0.4815 | 0.2500 | 0.2500 | 1.0000 | 3/3 |
| 4 | mid_agent_eval_v4_mismatch_probe (`mid_agent_eval_v4_mismatch_probe.json`) | token_prob | 12/12 | 0.0000 | — | 0.3550 | 0.5247 | 0.5247 | 2/12 |
| 5 | deepseek-chat (`api_probe_deepseek_2026-10-04.json`) | self_consistency | 6/12 | 0.2500 | 0.5000 | 0.5000 | 0.5000 | 1.0000 | 3/3 |
| 6 | mid_agent_eval_v2 (`mid_agent_eval_v2.json`) | token_prob | 12/12 | 0.0000 | — | 0.9999 | 0.9999 | 0.9999 | 12/12 |
| 7 | mid_agent_eval_v4 (`mid_agent_eval_v4.json`) | token_prob | 12/12 | 0.0000 | — | 0.9999 | 0.9999 | 0.9999 | 12/12 |
| 8 | mid_agent_eval_v4_trilogy (`mid_agent_eval_v4_trilogy.json`) | token_prob | 12/12 | 0.0000 | — | 0.9999 | 0.9999 | 0.9999 | 12/12 |

Caveats:

- n is 12 tasks per artifact: rankings are directional, not significant
- auroc is null when a run is all-correct or all-wrong (one class carries no ranking signal)
- conf sources differ (token_prob vs self_consistency); compare within a conf_type before across
- rows with a null conf (no final answer) count toward accuracy only; 'scored' shows how many rows the calibration metrics cover

results_sha256: `f4f4d16d8c41eccb9815b78e2eeb5c36430deca1a8e133ca288a547705ef8ad9`
