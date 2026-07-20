# Vibe-Trading Absorption Notes

Generated: 2026-06-28

Scope: research only. Do not connect broker accounts, do not place orders, and do not let any external agent update allocator defaults without the existing promotion gates.

## Source

- Repository: https://github.com/HKUDS/Vibe-Trading
- Relevant ideas reviewed from README and project metadata:
  - natural-language research task orchestration
  - Alpha factor zoo and backtest workflow
  - shadow account / real-trade-vs-rule comparison
  - MCP-style tool exposure
  - multi-agent debate and report generation

## What To Absorb

| Vibe-Trading idea | Local equivalent | Action |
| --- | --- | --- |
| Natural-language task to research report | CLI commands and daily pipeline | Keep CLI-first. Add thin task templates only after outputs are stable. |
| Alpha factor zoo | `factor-lab`, `theme-state`, `industry-chain-factor`, `berkshire-quality-factor` | Add factor families as sidecar studies, never direct allocation signals. |
| Shadow account | `paper-account`, `live-shadow`, `broker_statement` | Use for read-only comparison between actual/manual targets and rule-generated targets. |
| Agent debate | allocator promotion/readiness/audit reports | Prefer deterministic scorecards before multi-agent text debate. |
| MCP tool layer | `mcp_server.py`, `docs/quant-mcp.md` | Expose mature read-only reports through MCP after schema is stable. |
| Automated broker operations | Not allowed in current project | Explicitly excluded. Research-only boundary remains. |

## Immediate Local Tasks

1. Register Vibe-Trading as an external-method reference, not a strategy source.
2. Add a sidecar checklist to model-build audit:
   - Does a new idea have a reproducible data source?
   - Does it produce a CSV/snapshot/report?
   - Is there a no-future-function test?
   - Is the output compared against current mainline before any promotion?
3. Extend shadow-account review later:
   - compare paper target changes with rule-generated targets;
   - mark manual-vs-model divergence;
   - keep broker action as `none`.
4. Use factor-zoo ideas as candidate families:
   - price-volume factors;
   - quality/fundamental factors;
   - industry-chain/bottleneck factors;
   - market-state filters.

## Rejection Rules

- Do not import broker connectors.
- Do not store API keys in the project.
- Do not let LLM-generated strategy code bypass walk-forward validation.
- Do not update default allocator from narrative evidence alone.
- Do not mix external code into the mainline unless it passes local tests and audit.

## Current Decision

Status: `reference_only`

Vibe-Trading is useful as a workflow and architecture reference. It is not added as a direct dependency, not used as a trading source, and not connected to execution.
