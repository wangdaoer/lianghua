import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

import build_data_panel
import build_high_return_universe
from apply_personal_trade_overlay import _load_rules


def test_rule_loader_uses_json_only_when_yaml_dependency_is_unavailable(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "rules.json"
    path.write_text(json.dumps({"selection": {"top_n": 10}}), encoding="utf-8")
    monkeypatch.setitem(sys.modules, "yaml", None)

    loaded = _load_rules(path)

    assert loaded["selection"]["top_n"] == 10


def test_invalid_yaml_error_is_not_hidden_by_json_fallback(tmp_path: Path) -> None:
    path = tmp_path / "rules.yaml"
    path.write_text("selection: [unclosed", encoding="utf-8")

    with pytest.raises(yaml.YAMLError):
        _load_rules(path)


def test_unreadable_stock_history_is_reported(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "000001.csv"
    path.write_text("broken", encoding="utf-8")
    monkeypatch.setattr(
        build_high_return_universe,
        "_read_csv",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(pd.errors.ParserError("bad csv")),
    )

    with pytest.warns(RuntimeWarning, match="000001.csv"):
        result = build_high_return_universe.score_one_file(path, None, 180)

    assert result is None


def test_unreadable_snapshot_is_reported(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "ths_hs_a_share_2026-07-14.csv"
    path.write_text("broken", encoding="utf-8")
    monkeypatch.setattr(
        build_data_panel,
        "_read_panel_csv",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(pd.errors.ParserError("bad csv")),
    )

    with pytest.warns(RuntimeWarning, match="ths_hs_a_share_2026-07-14.csv"):
        with pytest.raises(ValueError, match="All selected snapshot files failed"):
            build_data_panel.build_panel((tmp_path,), None, None)
