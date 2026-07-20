# quant-mcp

`quant-mcp` is a read-only MCP server for this local quant research workspace.

It is intentionally limited to inspection tasks:

- `get_project_status`: inspect Git branch, remote, current commit, and working tree changes.
- `list_strategy_configs`: list YAML strategy configs under `configs/`.
- `read_strategy_config`: read and parse one YAML config under `configs/`.
- `get_latest_run_status`: read bounded previews from recent status and run-summary files.
- `search_project_outputs`: search bounded text snippets in `README.md`, `docs/`, `outputs/`, and root `tmp_*` files.

It does not trade, write files, delete files, update configs, run pipelines, or read entire data/output trees.

## Install dependencies

```powershell
cd D:\codex\量化
python -m pip install -r requirements.txt
```

## Run locally

```powershell
cd D:\codex\量化
python -m quant_etf_lab.mcp_server
```

The server uses STDIO transport, so the command will wait for an MCP client.

## Codex configuration

Codex MCP configuration lives in `config.toml`. You can place this in the global file
`C:\Users\86176\.codex\config.toml`, or in a trusted project-scoped file at
`D:\codex\量化\.codex\config.toml`.

```toml
[mcp_servers.quant_mcp]
command = "python"
args = ["-m", "quant_etf_lab.mcp_server"]
cwd = "D:\\codex\\量化"
startup_timeout_sec = 20
tool_timeout_sec = 60
enabled = true
default_tools_approval_mode = "auto"

[mcp_servers.quant_mcp.env]
QUANT_MCP_PROJECT_ROOT = "D:\\codex\\量化"
```

After editing the config, restart Codex or open a new Codex session, then check active MCP
servers with the MCP server list command available in your Codex surface.
