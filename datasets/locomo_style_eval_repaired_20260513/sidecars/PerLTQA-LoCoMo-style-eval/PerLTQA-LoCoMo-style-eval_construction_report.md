# Construction Report: PerLTQA-LoCoMo-style-eval

Generated at: 2026-05-13T12:12:45

## Status

Bootstrap harness artifact. Model-based expansion, source-entailment verification beyond heuristic seed checks, and human audit are not complete.

Construction mode: no-model deterministic conversion/seed QA. Model calls: 0.

## Summary

```json
{
  "samples": 10,
  "sessions": {
    "count": 10,
    "min": 25,
    "max": 31,
    "mean": 26.3,
    "total": 263
  },
  "turns": {
    "count": 10,
    "min": 202,
    "max": 262,
    "mean": 225.7,
    "total": 2257
  },
  "qa": {
    "count": 10,
    "min": 34,
    "max": 34,
    "mean": 34.0,
    "total": 340
  },
  "categories": {
    "1": 100,
    "2": 180,
    "3": 20,
    "4": 20,
    "5": 20
  }
}
```

## Notes

- PerLTQA PlanMode D bootstrap: selected 10 perltmem records with the most usable dialogue/event anchors; memory_anchor_turns are used for original event facts; profile, social_relationship, events, dialogues, and original QA are recorded in the fact ledger; original PerLTQA QA is not copied into final eval; non-protagonist source speakers are mapped to speaker_b in primary JSON and preserved as source_speaker in provenance.
