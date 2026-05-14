# Construction Report: OPELA-LoCoMo-style-eval

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
    "min": 4,
    "max": 8,
    "mean": 5.4,
    "total": 54
  },
  "turns": {
    "count": 10,
    "min": 65,
    "max": 335,
    "mean": 115.1,
    "total": 1151
  },
  "qa": {
    "count": 10,
    "min": 22,
    "max": 22,
    "mean": 22.0,
    "total": 220
  },
  "categories": {
    "1": 106,
    "2": 42,
    "3": 32,
    "4": 20,
    "5": 20
  }
}
```

## Notes

- OPELA PlanMode C bootstrap: selected top 10 rows by total_turn; turn order is reconstructed from aggregated per-speaker text columns and recorded in provenance.
