# LoadGuard MCP Server

Exposes LoadGuard's pipeline as **MCP tools** so IBM Bob can drive it during
development — computing load scores, generating plans, and running benchmarks
directly from a Bob session.

## Setup

From the repository root, install the optional MCP dependency and run the self-test:

```bash
pip install ".[bob]"
python mcp_server/server.py --self-test   # verify tools work without a Bob session
```

The server is intended for trusted local clients only. The MCP server itself does not provide authentication or network access controls; run it locally and do not expose it to untrusted networks.

## Tools

| Tool | What it returns |
| --- | --- |
| `compute_load_score(events)` | Cognitive Load Score (0-100) + level + factors |
| `analyze_workload(events, tasks)` | features, load report, plan, and impact |
| `benchmark_workload(events, tasks)` | objective before/after metrics |
| `propose_plan(events, tasks, history?)` | Granite proposal (gated) + guardian + baseline + trend + impact |
| `approve_plan(events, tasks, decision, feedback?)` | record a human accept / reject / edit in the audit trail |
| `export_plan_ics(events, tasks, start_epoch?)` | the protected focus/recovery blocks as `.ics` |
| `export_plan_csv(events, tasks)` | the resequenced task list as CSV |
| `pilot_evaluation(events, tasks, outcome_events?)` | baseline / projected / observed measurement |

## Register with IBM Bob

Add the server to [`.bob/mcp.json`](../.bob/mcp.json):

```json
{
  "mcpServers": {
    "loadguard": {
      "command": "python",
      "args": ["mcp_server/server.py"]
    }
  }
}
```

## Data format

`events` is a list of `{"timestamp": "...", "kind": "...", "duration_minutes": n, "meta": {...}}`. Timestamps accept the formats supported by the LoadGuard event parser.
`tasks` is a list of `{"title": "...", "priority": 1..5, ...}`. See
`demo/sample_events.jsonl` and `src/loadguard/sample_data.py` for examples.
