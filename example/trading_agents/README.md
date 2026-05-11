# Trading Agents Example

Sequential multi-agent research workflow:

```text
analyst -> researcher_bear -> researcher_bull -> trader -> risk_pm
```

The analyst can call `alpha_vantage_global_quote`, implemented as a
`FlowToolCall` in `tools.py`. Each role receives the output `Session` from the
previous role, so the workflow accumulates assistant chunks, tool results, and
workspace file writes.

Run from the repository root:

```bash
python example/trading_agents/main.py \
  --ticker NVDA \
  --as-of 2026-05-11 \
  --workdir .workspace/trading-agents
```

Requires OpenAI-compatible LLM settings. Set `ALPHA_VANTAGE_API_KEY` for market
data access.
