"""Research-only stock dependency network lab."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


DEFAULT_DATA_DIR = Path("data/processed/stocks")
DEFAULT_OUTPUT_DIR = Path("outputs/research/network_lab_latest")


@dataclass(frozen=True)
class NetworkLabResult:
    output_dir: Path
    edge_path: Path
    mst_path: Path
    cluster_summary_path: Path
    node_metrics_path: Path
    snapshot_path: Path
    report_path: Path
    snapshot: dict[str, Any]
    returns: pd.DataFrame
    edges: pd.DataFrame
    mst_edges: pd.DataFrame
    cluster_summary: pd.DataFrame
    node_metrics: pd.DataFrame


def _resolve(project_root: Path, path: str | Path | None, default: Path) -> Path:
    raw = Path(path) if path is not None else default
    return raw if raw.is_absolute() else project_root / raw


def _parse_date(value: Any) -> pd.Timestamp | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if len(text) == 8 and text.isdigit():
        parsed = pd.to_datetime(text, format="%Y%m%d", errors="coerce")
    else:
        parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed)


def _clean_code(value: Any) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits[-6:].zfill(6) if digits else text


def _csv_paths(data_dir: Path, recursive: bool = False) -> list[Path]:
    pattern = "**/*.csv" if recursive else "*.csv"
    return sorted(path for path in data_dir.glob(pattern) if path.is_file())


def _load_symbols_file(path: str | Path | None) -> list[str] | None:
    if path in (None, ""):
        return None
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Symbols file not found: {resolved}")
    frame = pd.read_csv(resolved, dtype={"code": str})
    if frame.empty:
        return []
    column = "code" if "code" in frame.columns else frame.columns[0]
    return [_clean_code(value) for value in frame[column].dropna().tolist()]


def _selected_paths(
    data_dir: Path,
    symbols: Iterable[str] | None = None,
    symbols_file: str | Path | None = None,
    max_symbols: int | None = None,
    recursive: bool = False,
) -> list[Path]:
    requested = [_clean_code(item) for item in (symbols or [])]
    file_symbols = _load_symbols_file(symbols_file)
    if file_symbols is not None:
        requested.extend(file_symbols)
    requested = list(dict.fromkeys(item for item in requested if item))
    paths = _csv_paths(data_dir, recursive=recursive)
    if requested:
        by_code = {path.stem[-6:].zfill(6): path for path in paths}
        selected = [by_code[code] for code in requested if code in by_code]
    else:
        selected = paths
    if max_symbols is not None and max_symbols > 0:
        selected = selected[: int(max_symbols)]
    return selected


def _read_price_csv(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"code": str})
    if frame.empty or "date" not in frame.columns or "close" not in frame.columns:
        raise ValueError(f"{path} must contain date and close columns.")
    data = frame.copy()
    if "code" not in data.columns:
        data["code"] = path.stem[-6:].zfill(6)
    if "name" not in data.columns:
        data["name"] = data["code"]
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["code"] = data["code"].map(_clean_code)
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    data = data.dropna(subset=["date", "code", "close"])
    data = data[data["close"] > 0].sort_values("date").drop_duplicates(subset=["date"], keep="last")
    return data[["date", "code", "name", "close"]].reset_index(drop=True)


def build_log_return_panel(
    data_dir: str | Path,
    symbols: Iterable[str] | None = None,
    symbols_file: str | Path | None = None,
    max_symbols: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    lookback_days: int | None = None,
    recursive: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build a wide date x symbol log-return panel from local close prices."""
    resolved_dir = Path(data_dir)
    paths = _selected_paths(resolved_dir, symbols=symbols, symbols_file=symbols_file, max_symbols=max_symbols, recursive=recursive)
    start = _parse_date(start_date)
    end = _parse_date(end_date)
    frames: list[pd.DataFrame] = []
    failures: list[dict[str, str]] = []
    names: dict[str, str] = {}
    for path in paths:
        try:
            prices = _read_price_csv(path)
            if start is not None:
                prices = prices[prices["date"] >= start]
            if end is not None:
                prices = prices[prices["date"] <= end]
            if lookback_days is not None and lookback_days > 0:
                prices = prices.tail(int(lookback_days) + 1)
            if len(prices) < 2:
                continue
            code = str(prices["code"].iloc[0])
            names[code] = str(prices["name"].iloc[-1])
            item = prices[["date", "close"]].copy()
            item[code] = np.log(item["close"] / item["close"].shift(1))
            frames.append(item[["date", code]].dropna())
        except (OSError, ValueError, pd.errors.EmptyDataError) as error:
            failures.append({"path": str(path), "error": str(error)})
    if not frames:
        panel = pd.DataFrame()
    else:
        panel = frames[0]
        for item in frames[1:]:
            panel = panel.merge(item, on="date", how="outer")
        panel = panel.sort_values("date").reset_index(drop=True)
    meta = {
        "data_dir": str(resolved_dir),
        "file_count": len(paths),
        "loaded_symbol_count": len([column for column in panel.columns if column != "date"]) if not panel.empty else 0,
        "row_count": int(len(panel)),
        "failures": failures[:20],
        "failure_count": len(failures),
        "symbol_names": names,
    }
    return panel, meta


def estimate_mutual_information(x: Iterable[float], y: Iterable[float], bins: int = 8) -> float:
    """Estimate mutual information from a two-dimensional histogram."""
    x_values = np.asarray(list(x), dtype=float)
    y_values = np.asarray(list(y), dtype=float)
    mask = np.isfinite(x_values) & np.isfinite(y_values)
    x_values = x_values[mask]
    y_values = y_values[mask]
    if len(x_values) < 3 or len(y_values) < 3:
        return 0.0
    if np.nanstd(x_values) == 0 or np.nanstd(y_values) == 0:
        return 0.0
    counts, _, _ = np.histogram2d(x_values, y_values, bins=max(int(bins), 2))
    total = counts.sum()
    if total <= 0:
        return 0.0
    pxy = counts / total
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    expected = px @ py
    mask = (pxy > 0) & (expected > 0)
    return float(np.sum(pxy[mask] * np.log(pxy[mask] / expected[mask])))


def _linear_information_from_correlation(correlation: float) -> float:
    rho = min(max(float(correlation), -0.999999), 0.999999)
    return float(-0.5 * np.log(max(1.0 - rho * rho, 1e-12)))


def build_network_edges(
    returns: pd.DataFrame,
    bins: int = 8,
    min_obs: int = 60,
) -> pd.DataFrame:
    """Build pairwise dependency edges from a log-return panel."""
    if returns.empty:
        return pd.DataFrame(
            columns=[
                "source",
                "target",
                "obs",
                "correlation",
                "correlation_distance",
                "mutual_information",
                "linear_information",
                "residual_mutual_information",
            ]
        )
    symbols = [column for column in returns.columns if column != "date"]
    rows: list[dict[str, Any]] = []
    for left_index, source in enumerate(symbols):
        for target in symbols[left_index + 1 :]:
            pair = returns[[source, target]].dropna()
            obs = int(len(pair))
            if obs < int(min_obs):
                continue
            x = pair[source].to_numpy(dtype=float)
            y = pair[target].to_numpy(dtype=float)
            if np.nanstd(x) == 0 or np.nanstd(y) == 0:
                continue
            correlation = float(np.corrcoef(x, y)[0, 1])
            if not np.isfinite(correlation):
                continue
            distance = float(np.sqrt(max(2.0 * (1.0 - correlation), 0.0)))
            mutual_information = estimate_mutual_information(x, y, bins=bins)
            linear_information = _linear_information_from_correlation(correlation)
            residual = max(float(mutual_information - linear_information), 0.0)
            rows.append(
                {
                    "source": source,
                    "target": target,
                    "obs": obs,
                    "correlation": correlation,
                    "correlation_distance": distance,
                    "mutual_information": mutual_information,
                    "linear_information": linear_information,
                    "residual_mutual_information": residual,
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=["source", "target", "obs", "correlation", "correlation_distance", "mutual_information", "linear_information", "residual_mutual_information"])
    return frame.sort_values(
        ["residual_mutual_information", "mutual_information", "correlation"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _find(parent: dict[str, str], item: str) -> str:
    while parent[item] != item:
        parent[item] = parent[parent[item]]
        item = parent[item]
    return item


def build_mst_edges(edges: pd.DataFrame, symbols: Iterable[str]) -> pd.DataFrame:
    """Build a minimum spanning tree using correlation distance."""
    symbols_list = list(dict.fromkeys(str(item) for item in symbols))
    if not symbols_list or edges.empty:
        return pd.DataFrame(columns=list(edges.columns) + ["network_role"])
    parent = {symbol: symbol for symbol in symbols_list}
    rank = {symbol: 0 for symbol in symbols_list}
    rows: list[dict[str, Any]] = []
    ranked_edges = edges.sort_values(["correlation_distance", "residual_mutual_information"], ascending=[True, False])
    for raw in ranked_edges.to_dict(orient="records"):
        source = str(raw.get("source"))
        target = str(raw.get("target"))
        if source not in parent or target not in parent:
            continue
        root_source = _find(parent, source)
        root_target = _find(parent, target)
        if root_source == root_target:
            continue
        if rank[root_source] < rank[root_target]:
            parent[root_source] = root_target
        elif rank[root_source] > rank[root_target]:
            parent[root_target] = root_source
        else:
            parent[root_target] = root_source
            rank[root_source] += 1
        row = dict(raw)
        row["network_role"] = "mst_core_edge"
        rows.append(row)
        if len(rows) >= max(len(symbols_list) - 1, 0):
            break
    columns = list(edges.columns)
    if "network_role" not in columns:
        columns.append("network_role")
    return pd.DataFrame(rows, columns=columns)


def summarize_network(
    symbols: Iterable[str],
    mst_edges: pd.DataFrame,
    name_lookup: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    symbols_list = list(dict.fromkeys(str(item) for item in symbols))
    parent = {symbol: symbol for symbol in symbols_list}
    rank = {symbol: 0 for symbol in symbols_list}
    degree = {symbol: 0 for symbol in symbols_list}
    residual_sum = {symbol: 0.0 for symbol in symbols_list}
    for row in mst_edges.to_dict(orient="records"):
        source = str(row.get("source"))
        target = str(row.get("target"))
        if source not in parent or target not in parent:
            continue
        root_source = _find(parent, source)
        root_target = _find(parent, target)
        if root_source != root_target:
            if rank[root_source] < rank[root_target]:
                parent[root_source] = root_target
            elif rank[root_source] > rank[root_target]:
                parent[root_target] = root_source
            else:
                parent[root_target] = root_source
                rank[root_source] += 1
        residual = float(row.get("residual_mutual_information") or 0.0)
        degree[source] += 1
        degree[target] += 1
        residual_sum[source] += residual
        residual_sum[target] += residual

    component_members: dict[str, list[str]] = {}
    for symbol in symbols_list:
        component_members.setdefault(_find(parent, symbol), []).append(symbol)

    name_lookup = name_lookup or {}
    cluster_rows = []
    for cluster_id, members in enumerate(sorted(component_members.values(), key=lambda item: (-len(item), item[0])), start=1):
        cluster_edges = mst_edges[mst_edges["source"].isin(members) & mst_edges["target"].isin(members)] if not mst_edges.empty else pd.DataFrame()
        cluster_rows.append(
            {
                "cluster_id": cluster_id,
                "symbol_count": len(members),
                "symbols": ",".join(members),
                "avg_correlation": float(pd.to_numeric(cluster_edges.get("correlation"), errors="coerce").mean()) if not cluster_edges.empty else np.nan,
                "avg_residual_mutual_information": float(pd.to_numeric(cluster_edges.get("residual_mutual_information"), errors="coerce").mean()) if not cluster_edges.empty else np.nan,
            }
        )

    node_rows = []
    for symbol in symbols_list:
        node_rows.append(
            {
                "code": symbol,
                "name": name_lookup.get(symbol, symbol),
                "mst_degree": int(degree.get(symbol, 0)),
                "avg_residual_mutual_information": residual_sum.get(symbol, 0.0) / degree[symbol] if degree.get(symbol, 0) else 0.0,
            }
        )
    clusters = pd.DataFrame(cluster_rows)
    nodes = pd.DataFrame(node_rows).sort_values(["mst_degree", "avg_residual_mutual_information", "code"], ascending=[False, False, True]).reset_index(drop=True)
    return clusters, nodes


def _render_report(snapshot: dict[str, Any], edges: pd.DataFrame, clusters: pd.DataFrame, nodes: pd.DataFrame) -> str:
    if edges.empty:
        edge_rows = "| N/A | N/A | N/A | N/A | N/A | N/A |"
    else:
        edge_rows = "\n".join(
            "| `{source}` | `{target}` | {corr:.3f} | {dist:.3f} | {mi:.4f} | {residual:.4f} |".format(
                source=row.get("source"),
                target=row.get("target"),
                corr=float(row.get("correlation") or 0.0),
                dist=float(row.get("correlation_distance") or 0.0),
                mi=float(row.get("mutual_information") or 0.0),
                residual=float(row.get("residual_mutual_information") or 0.0),
            )
            for _, row in edges.head(10).iterrows()
        )
    if nodes.empty:
        node_rows = "| N/A | N/A | N/A | N/A |"
    else:
        node_rows = "\n".join(
            "| `{code}` | {name} | {degree} | {residual:.4f} |".format(
                code=row.get("code"),
                name=row.get("name"),
                degree=int(row.get("mst_degree") or 0),
                residual=float(row.get("avg_residual_mutual_information") or 0.0),
            )
            for _, row in nodes.head(10).iterrows()
        )
    return f"""# Network Lab Report

Generated at: `{snapshot.get("generated_at")}`

This is a research-only dependency-network report. It uses local historical close prices to study correlation distance, mutual information, residual mutual information, and MST structure. It does not connect to brokers, place orders, or provide investment advice.

## Summary

| Item | Value |
| --- | ---: |
| Loaded symbols | {snapshot.get("loaded_symbol_count", 0)} |
| Return rows | {snapshot.get("return_row_count", 0)} |
| Edge count | {snapshot.get("edge_count", 0)} |
| MST edge count | {snapshot.get("mst_edge_count", 0)} |
| Top residual MI | {snapshot.get("top_residual_mutual_information", 0.0):.4f} |
| Top MI pair | `{snapshot.get("top_mutual_information_pair", "N/A")}` |
| Start / end | {snapshot.get("start_date", "N/A")} / {snapshot.get("end_date", "N/A")} |
| Broker action | `{snapshot.get("broker_action")}` |

## Top Hidden-Linkage Edges

| Source | Target | Corr | Distance | MI | Residual MI |
| --- | --- | ---: | ---: | ---: | ---: |
{edge_rows}

## Core MST Nodes

| Code | Name | Degree | Avg residual MI |
| --- | --- | ---: | ---: |
{node_rows}

## Files

- Edges CSV: `{snapshot.get("edge_path")}`
- MST CSV: `{snapshot.get("mst_path")}`
- Cluster summary CSV: `{snapshot.get("cluster_summary_path")}`
- Node metrics CSV: `{snapshot.get("node_metrics_path")}`
- Snapshot JSON: `{snapshot.get("snapshot_path")}`
"""


def run_network_lab(
    project_root: str | Path = Path("."),
    data_dir: str | Path | None = DEFAULT_DATA_DIR,
    output_dir: str | Path | None = DEFAULT_OUTPUT_DIR,
    symbols: Iterable[str] | None = None,
    symbols_file: str | Path | None = None,
    max_symbols: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    lookback_days: int | None = 120,
    top_edges: int = 50,
    bins: int = 8,
    min_obs: int = 60,
    recursive: bool = False,
) -> NetworkLabResult:
    root = Path(project_root).resolve()
    resolved_data = _resolve(root, data_dir, DEFAULT_DATA_DIR)
    resolved_output = _resolve(root, output_dir, DEFAULT_OUTPUT_DIR)
    returns, meta = build_log_return_panel(
        resolved_data,
        symbols=symbols,
        symbols_file=symbols_file,
        max_symbols=max_symbols,
        start_date=start_date,
        end_date=end_date,
        lookback_days=lookback_days,
        recursive=recursive,
    )
    if returns.empty:
        raise ValueError(f"No log-return rows built from {resolved_data}")
    loaded_symbols = [column for column in returns.columns if column != "date"]
    edges = build_network_edges(returns, bins=bins, min_obs=min_obs)
    if edges.empty:
        raise ValueError("No network edges built; lower --min-obs or provide more overlapping history.")
    mst_edges = build_mst_edges(edges, loaded_symbols)
    clusters, nodes = summarize_network(loaded_symbols, mst_edges, name_lookup=meta.get("symbol_names") or {})
    top_edges_frame = edges.head(max(int(top_edges), 1)).copy()

    resolved_output.mkdir(parents=True, exist_ok=True)
    edge_path = resolved_output / "network_edges.csv"
    mst_path = resolved_output / "network_mst_edges.csv"
    cluster_summary_path = resolved_output / "cluster_summary.csv"
    node_metrics_path = resolved_output / "node_metrics.csv"
    snapshot_path = resolved_output / "network_lab_snapshot.json"
    report_path = resolved_output / "network_lab.md"

    top_mi = edges.sort_values("mutual_information", ascending=False).iloc[0]
    top_residual = edges.sort_values("residual_mutual_information", ascending=False).iloc[0]
    snapshot = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_dir": str(resolved_data),
        "output_dir": str(resolved_output),
        "loaded_symbol_count": int(len(loaded_symbols)),
        "return_row_count": int(len(returns)),
        "edge_count": int(len(edges)),
        "top_edges": int(top_edges),
        "mst_edge_count": int(len(mst_edges)),
        "cluster_count": int(len(clusters)),
        "bins": int(bins),
        "min_obs": int(min_obs),
        "lookback_days": int(lookback_days) if lookback_days is not None else None,
        "start_date": str(pd.to_datetime(returns["date"], errors="coerce").min().date()) if not returns.empty else None,
        "end_date": str(pd.to_datetime(returns["date"], errors="coerce").max().date()) if not returns.empty else None,
        "failure_count": meta.get("failure_count", 0),
        "failures": meta.get("failures", []),
        "top_mutual_information_pair": f"{top_mi['source']}-{top_mi['target']}",
        "top_mutual_information": float(top_mi["mutual_information"]),
        "top_residual_mutual_information_pair": f"{top_residual['source']}-{top_residual['target']}",
        "top_residual_mutual_information": float(top_residual["residual_mutual_information"]),
        "edge_path": str(edge_path),
        "mst_path": str(mst_path),
        "cluster_summary_path": str(cluster_summary_path),
        "node_metrics_path": str(node_metrics_path),
        "snapshot_path": str(snapshot_path),
        "report_path": str(report_path),
        "research_only": True,
        "broker_action": "none",
        "note": "Candidate-generation evidence only; validate through outcome review, walk-forward, and allocator/risk-budget gates before promotion.",
    }

    top_edges_frame.to_csv(edge_path, index=False, encoding="utf-8")
    mst_edges.to_csv(mst_path, index=False, encoding="utf-8")
    clusters.to_csv(cluster_summary_path, index=False, encoding="utf-8")
    nodes.to_csv(node_metrics_path, index=False, encoding="utf-8")
    snapshot_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(_render_report(snapshot, top_edges_frame, clusters, nodes), encoding="utf-8")

    return NetworkLabResult(
        output_dir=resolved_output,
        edge_path=edge_path,
        mst_path=mst_path,
        cluster_summary_path=cluster_summary_path,
        node_metrics_path=node_metrics_path,
        snapshot_path=snapshot_path,
        report_path=report_path,
        snapshot=snapshot,
        returns=returns,
        edges=top_edges_frame,
        mst_edges=mst_edges,
        cluster_summary=clusters,
        node_metrics=nodes,
    )
