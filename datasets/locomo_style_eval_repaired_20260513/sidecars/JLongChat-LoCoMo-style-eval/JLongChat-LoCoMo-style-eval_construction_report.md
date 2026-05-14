# Construction Report: JLongChat-LoCoMo-style-eval

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
    "min": 5,
    "max": 21,
    "mean": 11.4,
    "total": 114
  },
  "turns": {
    "count": 10,
    "min": 60,
    "max": 249,
    "mean": 117.5,
    "total": 1175
  },
  "qa": {
    "count": 10,
    "min": 22,
    "max": 22,
    "mean": 22.0,
    "total": 220
  },
  "categories": {
    "1": 100,
    "2": 48,
    "3": 32,
    "4": 20,
    "5": 20
  }
}
```

## Notes

- JLongChat PlanMode A/B bootstrap: 4 LAC rooms plus 6 JMSC pairs selected; no Japanese source text was translated or polished; multi-party LAC rooms are reduced to two loader speakers in primary JSON while raw source speakers are preserved as source_speaker in provenance.
