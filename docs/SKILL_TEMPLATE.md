# HeliosLM Skill Template Specification

Adapted from [`dslsdzc/rev-skills`](https://github.com/dslsdzc/rev-skills)
`docs/skill-template.md` (skill documentation: CC BY 4.0, attribution
required and preserved here). HeliosLM adaptations: English-only docs,
skills rooted at `skills/`, validation by `validate_skills.py` (pure
stdlib, gated in CI by `evastar/tests/test_skill_docs.py`).

## Layout

Each skill is a directory under `skills/`:

```
skills/<skill-name>/
├── SKILL.md            # required
└── references/         # optional: deep knowledge, probes, gotchas
```

Three tiers, mirroring rev-skills' entry → gateway → atomic routing:

- **entry** — exactly one (`skills/agent-entry/`): probe the task, identify
  the domain, dispatch to a gateway. Never does domain work itself.
- **gateway** — one per domain: full-workflow overview + a selection tree
  over its atomic skills.
- **atomic** — one concrete capability. Default tier.

## Frontmatter (YAML, required)

| key | rule |
|---|---|
| `name` | must equal the directory name |
| `type` | `entry` / `gateway` / `atomic` (default: `atomic`) |
| `description` | must state trigger conditions with "Use when ..." (validator-enforced) |
| `guard` | optional JSON object, **required for sensitive capabilities**: `{"require_authorization": true, "forbidden": ["<action label>", ...]}` |

`guard` semantics: machine-readable safety precondition. When
`require_authorization` is true, the entry skill must establish the
authorization level before dispatch — `owned` / `eval-benchmark` /
`external-unknown`; for `external-unknown` only read-only analysis is
allowed and the user is asked about ownership first. This mirrors
rev-skills' `RE_AUTH` gating and complements HeliosLM's runtime trust
gate (which guards *numerical* confidence, not *authorization*).

## Atomic skill sections (fixed order)

1. frontmatter
2. `# Title`
3. `## Intent classifier` — optional but recommended for multi-branch
   skills: user-intent → procedure-path mapping table. Multi-branch
   skills should add an **input inventory** (which assets must already
   exist) and a **decision gate** (input-form decision tree for
   dispatch). Single-task skills may omit all three.
4. `## When to use / when not to use` — required; explicit boundaries
   prevent mis-triggering.
5. `## Tool preparation` — required whenever the procedure needs tools:
   install command per environment + a verify command for each tool.
6. `## Procedure` — required. Numbered, executable steps; never "handle
   the rest similarly". **Knowledge/operation separation**: mechanism
   rationale stays in the procedure; volatile parameters (offsets,
   version-dependent details) live in `references/probes.md` and are
   probed at runtime. Complex skills use the four-layer split:
   - *Method Overview* — one sentence per step: goal / essence
   - *Operational Checklist* — numbered actions and outputs only
   - *Failure Modes* — pitfalls (here + `references/gotchas.md`)
   - *Implementation Notes* — experience in
     `references/implementation-notes.md`; steps carry only a one-line
     `see [[implementation-notes]]` reference
7. `## Cross-domain` — compound tasks that reference this skill, with
   `[[skill-name]]` links.
8. `## Pitfalls` — required, **at least 3** concrete entries.

## Gateway skill sections (fixed order)

1. frontmatter (`description` declares the gateway role)
2. `# Title`
3. `## Full workflow` — the domain end-to-end, numbered
4. `## Which atomic skill when` — selection tree by input features/goal
5. `## Cross-domain`
6. `## Pitfalls`

## Linking rules (validator-enforced)

- Body is plain Markdown; no tool-private syntax; frontmatter is standard
  YAML.
- `[[skill-name]]` links a sibling skill (directory under `skills/`).
- `[[file]]` links a file in the skill's own `references/` (extension
  optional).
- Cross-skill references links **must** be prefixed:
  `[[skill-name/file]]` — bare names are ambiguous because filenames like
  `gotchas.md` recur across skills.
- Placeholders use angle brackets — `[[<skill-name>]]`, `[[<file>]]` —
  which the validator never resolves, so documentation examples cannot
  create dead links.
- Links are validated in `SKILL.md` **and** in every `references/*.md`.

## Validation

```bash
python validate_skills.py        # repo root; exit 1 on any violation
```

CI runs the same check through `evastar/tests/test_skill_docs.py`
(the pytest step collects it; the `test_v5.py` runner does not). A
repository with no `skills/` directory passes vacuously — the gate
becomes load-bearing the moment the first skill lands.
