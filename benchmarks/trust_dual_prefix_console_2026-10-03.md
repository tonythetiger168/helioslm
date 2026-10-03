# Trust dual-prefix run console record (2026-10-03, pre-v5.37l harness)

Raw console output archived from the user's v5.37k run. Headline:
- ungrounded_p_mean 0.0228 / grounded_p_mean 0.1904 -- 8.4x contrast
- tiers emerged: high ~0.71, medium ~0.46-0.48, low <0.3 in one run
- 4 DIRECT [grounded] tasks 4/4 WRONG (finals -48 / YeY9W999 / 71 / 1)
  -> the harness executed them UNTREATED; trust conditioned on grounded
  outcomes must be paired with the grounding intervention. This
  motivated the TherapyPair composition (v5.37l)
- v5 rerun: premature-finish refusal fired at write_read task 10
  (9/9 calc/str correct before it), then AbstainOracle lacked decide()
  and the harness died -- refusal correct, assembly incomplete

Key console lines (abridged):
[grounded] Task correct=False abstained=False p_min=0.705 final='YeY9W999'
[grounded] Task correct=False abstained=False p_min=0.456 final='-48'
ungrounded_p_mean: 0.0228, grounded_p_mean: 0.1904, abstained 20/24
