"""A-share trading-day and after-close data readiness gate."""

from __future__ import annotations

from datetime import date, datetime, time
from typing import Any

import pandas as pd


AFTER_CLOSE_CUTOFF = time(15, 30)


def _parse_date(value: Any) -> date | None:
    if value in (None, ""):
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).date()


def _parse_datetime(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).to_pydatetime()


def build_a_share_trading_gate(
    snapshot: dict[str, Any],
    as_of: str | date,
    generated_at: str | datetime | None = None,
    after_close_cutoff: time = AFTER_CLOSE_CUTOFF,
) -> dict[str, Any]:
    """Classify whether an after-close daily run can interpret same-day data.

    The gate is deliberately local-first. Weekends are deterministic non-trading
    days; weekdays are treated as expected trading days unless same-day local
    equity data proves readiness. Holiday edge cases therefore stay visible as
    data-not-ready checks instead of being guessed.
    """

    as_of_date = _parse_date(as_of)
    if as_of_date is None:
        return {
            "trading_day_gate_status": "invalid_as_of_date",
            "is_a_share_trading_day": None,
            "trading_day_evidence": "invalid_as_of_date",
            "after_close_cutoff": after_close_cutoff.strftime("%H:%M"),
            "after_close_data_status": "unknown",
            "trading_day_gate_reason": "as_of_date_could_not_be_parsed",
            "trading_day_gate_action": "fix_as_of_date",
        }

    run_at = generated_at if isinstance(generated_at, datetime) else _parse_datetime(generated_at)
    if run_at is None:
        run_at = datetime.now()
    latest_local = _parse_date(snapshot.get("local_equity_latest_date"))
    weekday = as_of_date.weekday()
    cutoff_reached = run_at.date() > as_of_date or (run_at.date() == as_of_date and run_at.time() >= after_close_cutoff)

    base = {
        "is_after_close_cutoff_reached": bool(cutoff_reached),
        "after_close_cutoff": after_close_cutoff.strftime("%H:%M"),
        "trading_day_gate_checked_at": run_at.isoformat(timespec="seconds"),
        "trading_day_gate_latest_local_equity_date": latest_local.isoformat() if latest_local else None,
    }

    if weekday >= 5:
        return {
            **base,
            "trading_day_gate_status": "non_trading_day",
            "is_a_share_trading_day": False,
            "trading_day_evidence": "weekend_rule",
            "after_close_data_status": "not_required",
            "trading_day_gate_reason": "as_of_date_is_weekend",
            "trading_day_gate_action": "skip_new_signal_interpretation",
        }

    if latest_local is None:
        return {
            **base,
            "trading_day_gate_status": "trading_day_data_missing",
            "is_a_share_trading_day": True,
            "trading_day_evidence": "weekday_rule_without_holiday_calendar",
            "after_close_data_status": "missing",
            "trading_day_gate_reason": "local_equity_latest_date_missing",
            "trading_day_gate_action": "refresh_local_market_data",
        }

    if latest_local > as_of_date:
        return {
            **base,
            "trading_day_gate_status": "future_dated_data",
            "is_a_share_trading_day": None,
            "trading_day_evidence": "local_data_date_after_as_of",
            "after_close_data_status": "future_dated",
            "trading_day_gate_reason": "local_equity_latest_date_is_after_as_of_date",
            "trading_day_gate_action": "inspect_local_data_dates",
        }

    if latest_local == as_of_date:
        return {
            **base,
            "trading_day_gate_status": "trading_day_data_ready",
            "is_a_share_trading_day": True,
            "trading_day_evidence": "local_data_same_date",
            "after_close_data_status": "ready",
            "trading_day_gate_reason": "local_equity_data_matches_as_of_date",
            "trading_day_gate_action": "continue_routine_review",
        }

    if not cutoff_reached:
        return {
            **base,
            "trading_day_gate_status": "before_close_wait",
            "is_a_share_trading_day": True,
            "trading_day_evidence": "weekday_rule_without_holiday_calendar",
            "after_close_data_status": "before_close",
            "trading_day_gate_reason": "after_close_cutoff_not_reached",
            "trading_day_gate_action": "wait_until_after_close",
        }

    return {
        **base,
        "trading_day_gate_status": "trading_day_data_not_ready",
        "is_a_share_trading_day": True,
        "trading_day_evidence": "weekday_rule_without_holiday_calendar",
        "after_close_data_status": "not_ready",
        "trading_day_gate_reason": "weekday_after_close_but_local_equity_data_is_not_current",
        "trading_day_gate_action": "verify_trading_day_or_refresh_data",
    }
