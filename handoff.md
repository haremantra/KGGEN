# Handoff — `claude/modality-checker-spec-MSHvh`

Status as of latest commit on this branch. For someone picking up cold.

## TL;DR

Four stacked layers shipped over 9 commits. All on
`claude/modality-checker-spec-MSHvh`, pushed, **no PR opened yet**.

```
  redline/    verdict assigner    FAITHFUL / MISSED / DRIFTED / PHANTOM / EXPECTED_SKIP
  polarity/   FM-B06 surface       within-clause + cross-clause signals
  modality/   deontic primitive    rules + diff + subject normalization
  ↑ pipeline integration + `python -m src.main modality` CLI
```

Branch is clean. `pytest tests/unit/` → **372 passing**, 11 pre-existing
offline-env failures unrelated to this work (HuggingFace embedding download
fails in the sandbox; same 11 failed before this work).

## Commits, oldest to newest

| # | SHA | What it did |
|---|---|---|
| 1 | `3728454` | `src/modality/` standalone module + 55-test executable spec. Synchronous regex-based deontic modality detector (OBLIGATION / PERMISSION / PROHIBITION / ENTITLEMENT). |
| 2 | `8482766` | Rule refinements (`IN NO EVENT SHALL` reversed-order PROHIBITION) + paragraph-break subject boundary + `confidence: float` → `strength: Literal["LOW","MEDIUM","HIGH"]`. |
| 3 | `0bf0059` | `ModalityChecker` wired into `ContractAnalysisPipeline`. `AnalyzedClause.modality_findings` field added (default empty list — backward compatible). |
| 4 | `44e3b12` | `src/polarity/` — decision-support detector mapping `(modality, subject)` patterns to FM-B06 (party misattribution) / FM-D02 (duplicate inconsistency) candidates. |
| 5 | `907a411` | `PolarityChecker` wired into `analyze_contract_file`. Return tuple extended `(analysis, risk, deps)` → `(analysis, risk, deps, polarity)`. Output JSON gains `polarity` key. `pdfplumber` import demoted to lazy on the PDF branch only. |
| 6 | `10a3bb1` | `ModalityChecker.check_diff(original, redlined)` — diff primitive returning `kept` / `added` / `removed` / `drifted: list[DriftPair]`. `normalize_subject` promoted from polarity-private to public `src.modality.normalize_subject`. |
| 7 | `b7d0296` | `python -m src.main modality <pdf>` CLI subcommand with `--with-classifier`, `--with-polarity`, `--show`, `-o`. `pdfplumber` and `neo4j` demoted to lazy proxies in `src/main.py` so the CLI module loads without those deps. |
| 8 | `8cc2865` | `src/redline/` — `RedlineVerdictAssigner` mapping `ModalityDiff` + `Recommendation[]` to verdicts in the observed-redline-analysis vocabulary. Skill-aligned `fidelity_score` and `confabulation_score` formulas. |
| 9 | `5e35720` | **Predicate-aware diff matching**: every finding gains `predicate_hint` (next ~3 lowercased tokens after the modal verb, sentence-bounded). `check_diff` matches on `(subject, modality, predicate_hint)` so two distinct statements that share `(subject, modality)` no longer collide as `kept`. Redline `_assign_modify` gains a (removed-before + added-expected) fallback. |

## Files added / changed

```
 src/api/routes.py                   |   1 +
 src/main.py                         | 160 ++++++++-
 src/modality/__init__.py            |  56 +++   NEW
 src/modality/checker.py             | 257 +++   NEW
 src/modality/rules.py               | 152 +++   NEW
 src/modality/types.py               | 114 +++   NEW
 src/pipeline.py                     |  49 ++-
 src/polarity/__init__.py            |  33 +++   NEW
 src/polarity/checker.py             | 248 +++   NEW
 src/polarity/types.py               | 108 +++   NEW
 src/redline/__init__.py             |  54 +++   NEW
 src/redline/assigner.py             | 293 +++   NEW
 src/redline/types.py                | 118 +++   NEW
 tests/unit/test_cli_modality.py     |  98 +++   NEW
 tests/unit/test_modality.py         | 673 +++   NEW
 tests/unit/test_pipeline_helpers.py | 219 +++
 tests/unit/test_polarity.py         | 365 +++   NEW
 tests/unit/test_redline_verdict.py  | 411 +++   NEW
 18 files changed, +3400 / −9
```

## How to run

### Default (offline, no API key, no embedding model)
```bash
python -m src.main modality data/sample_software_license.txt --show 30
python -m src.main modality contract.pdf -o report.json
```

### With classifier (per-clause CUAD labels)
Requires `ANTHROPIC_API_KEY` and a downloaded sentence-transformers model.
```bash
python -m src.main modality contract.pdf --with-classifier
```

### With polarity (FM-B06 / FM-D02 surfaces)
Implies `--with-classifier`.
```bash
python -m src.main modality contract.pdf --with-polarity
```

### Programmatic redline-compliance (memo vs redline)
```python
from src.modality import ModalityChecker, Modality
from src.redline import RedlineVerdictAssigner, Recommendation

diff = ModalityChecker().check_diff(original_text, redlined_text)
recs = [
    Recommendation(
        rec_id="REC-001", rec_type="MODIFY",
        before_modality=Modality.OBLIGATION, before_subject="Licensor",
        expected_modality=Modality.PERMISSION, expected_subject="Licensor",
        selected=True,
    ),
    # ...
]
report = RedlineVerdictAssigner().assign(diff, recs, contract_id="x")
print(report.fidelity_score, report.confabulation_score, report.counts_by_verdict)
```

## Architectural decisions worth knowing

### Categorical, not numeric
`ModalFinding.strength` is `Literal["LOW","MEDIUM","HIGH"]`, not a float.
Matches the convention in `src/extraction/extractor.py:CONFIDENCE_MAP`.
The numeric scores looked like a posterior, but were just a 3-bucket
lookup; renaming + retyping makes the semantics honest.

### `normalize_subject` is public, in modality
It originally lived as `_normalize_subject` inside `src/polarity/checker.py`.
When `check_diff` needed it, I promoted it. Polarity now imports it via:
`from ..modality import normalize_subject as _normalize_subject` (the
private-name alias preserves the existing test imports).

### Lazy proxies in `src/main.py`
`pdfplumber` and `neo4j-driver` are now lazy-loaded from inside helper
functions of the same name. Net effect: text-only subcommands
(`modality`, `classify`, `search`, etc.) load and run on machines where
those optional deps aren't installed. All 7 existing call sites are
unchanged at the call-site name.

### Predicate-aware diff matching
This was the headline correctness fix. Without it, two distinct
statements like `"Licensee shall pay"` and `"Licensee shall indemnify"`
collapsed as a single `kept` match, and the redline-verdict layer
returned wrong verdicts. The fix: capture a `predicate_hint` per finding
and include it in the diff match keys.

### `analyze_contract_file` returns a 4-tuple
```python
(ContractAnalysis, RiskAssessment | None,
 InterdependencyReport | None, PolarityReport | None)
```
Only one external caller (`src/main.py:cmd_analyze`) was affected;
updated.

## Known limitations (honest punch list)

1. **Subject extraction looks backward only.** Sentences like
   `"Under no circumstances shall Licensee assign rights"` produce
   `subject=None` because the extractor scans tokens *before* the modal
   verb. For reversed-order prohibition patterns, the subject lives
   *after*. ~30 min fix to add a forward-look fallback when
   `modal_phrase` matches one of the reversed prohibition rules.

2. **`predicate_hint` doesn't help when the predicate also changes.**
   `"Licensor shall indemnify Licensee"` → `"Licensee shall indemnify
   Licensor"` is a party inversion with object swap. The predicate
   hints (`"indemnify licensee"` vs `"indemnify licensor"`) differ, so
   the diff classifies as remove+add rather than SUBJECT-drifted. The
   redline verdict layer's MODIFY fallback recovers it correctly, but
   no explicit DriftKind.SUBJECT is emitted. A predicate-similarity
   match (e.g. token overlap) could improve this.

3. **No memo parser.** `Recommendation` instances must be
   caller-constructed. End-to-end use against a real memo doc requires
   a parser of some agreed format (JSON schema + loader).

4. **No event-sourced emission.** The observed-redline-analysis skill
   describes a coordinator/harness pattern with JSONL audit logs.
   We have the analytical layers; we don't have the coordinator.

5. **No self-review checks.** Silent-failure audit, deletion
   verification, scalar verification, denominator validation — all
   defined in the skill but not implemented.

6. **Polarity layer ships only 2 of N possible signal kinds.**
   `MUTUAL_DRIFT` and `DUPLICATE_INCONSISTENCY` are decision-support
   only. A static "expected direction" map per CUAD label (e.g.
   `Cap On Liability` typically caps the seller) would let polarity
   issue stronger calls — but it requires per-label legal judgment.

## Next-action ranking (recommended order)

1. **Fix subject extraction for reversed-order modals** — small, ~30 min,
   tightens FM-B06 detection. Concrete test scenario already exists in
   `data/sample_software_license.txt`.
2. **Memo parser** — pick a JSON schema for memo recs; write a loader.
   Makes the redline pipeline runnable end-to-end against a real memo.
3. **Self-review checks** in `src/redline/` — cheap quality wins per
   the skill's Step 7 (silent-failure audit, deletion verification,
   scalar verification, denominator validation).
4. **Event-sourced emission** mirroring the observed-redline coordinator
   — bigger architectural slice; defer until use case demands it.
5. **Predicate-similarity matching** in `check_diff` — handles the
   party-inversion-with-object-swap case as a SUBJECT drift instead of
   remove+add. Diminishing returns; the verdict layer already recovers
   most of these via fallback.

## Things deliberately not done

- No PR opened on GitHub. (You didn't ask for one.)
- No edits to `src/risk/`, `src/interdependency/`, `src/resolution/`,
  `src/portfolio/`, or `streamlit_app.py`. They're read-only consumers
  of the new modality data and don't need updates.
- No changes to existing test fixtures except where forced
  (one test in `test_modality.py` updated to reflect the
  predicate-aware behavior change).
- No new top-level deps added to `pyproject.toml`. Everything uses
  Python stdlib (`re`, `dataclasses`, `enum`, `typing`).

## Verifying the branch from scratch

```bash
git checkout claude/modality-checker-spec-MSHvh
pytest tests/unit/ --no-cov -q
# Expect: 372 passing, 11 pre-existing failures (test_aggregation,
# test_resolution, test_search_service — all OSError on HuggingFace).

# Smoke the CLI:
python -m src.main modality data/sample_software_license.txt --show 5

# Smoke the redline layer (writes to stdout, no file I/O):
python - <<'PY'
from src.modality import ModalityChecker, Modality
from src.redline import RedlineVerdictAssigner, Recommendation

diff = ModalityChecker().check_diff(
    "Licensor shall indemnify Licensee against third-party claims.",
    "Licensee shall indemnify Licensor against third-party claims.",
)
rec = Recommendation(
    rec_id="REC-POL", rec_type="MODIFY",
    before_modality=Modality.OBLIGATION, before_subject="Licensor",
    expected_modality=Modality.PERMISSION, expected_subject="Licensor",
    selected=True,
)
report = RedlineVerdictAssigner().assign(diff, [rec])
print(report.verdicts[0].verdict.value)  # expect: DRIFTED
PY
```
