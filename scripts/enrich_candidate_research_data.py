"""Enrich current stock candidates with public fundamental and industry-chain data.

The script is research-only. It fetches public data through AKShare, writes raw
responses for audit, then builds the CSV inputs consumed by
berkshire-quality-factor and industry-chain-factor.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import akshare as ak
import pandas as pd


FUNDAMENTAL_COLUMNS = [
    "code",
    "name",
    "sector_type",
    "listed_years",
    "roe_10y_avg",
    "fcf_5y_cumulative",
    "interest_coverage",
    "gross_margin_5y_avg",
    "ocf_to_net_income_5y_avg",
    "net_margin_10y_avg",
    "share_dilution_5y",
    "recent_ocf_positive",
    "net_margin_recovery",
    "business_model_type",
    "research_notes",
]
QUALITATIVE_COLUMNS = [
    "code",
    "circle_of_competence_score",
    "moat_score",
    "management_score",
    "safety_margin_score",
    "thesis_clarity_score",
    "red_flag_count",
    "data_confidence",
    "qualitative_notes",
]
CHAIN_COLUMNS = [
    "code",
    "name",
    "industry_chain",
    "chain_node",
    "node_role",
    "demand_certainty",
    "supply_constraint",
    "attention_gap",
    "value_capture",
    "catalyst_strength",
    "dilution_risk",
    "financing_risk",
    "substitution_risk",
    "crowding_risk",
    "valuation_risk",
    "upstream_dependency",
    "customer_validation",
    "qualification_cycle",
    "pricing_power",
    "research_notes",
]

SECTOR_FINANCIAL_KEYWORDS = ("银行", "金融", "证券", "保险", "农商行")
SECTOR_POWER_KEYWORDS = ("电力", "核电", "能源", "煤", "神华", "发电")
SECTOR_CONSUMER_KEYWORDS = ("食品", "调味", "肉", "速冻", "消费")
SECTOR_TECH_KEYWORDS = ("半导体", "芯片", "电子", "光电", "集成电路", "材料", "封装", "显示", "光刻")

CHAIN_OVERRIDES: dict[str, dict[str, Any]] = {
    "000823": {
        "industry_chain": "pcb_display_electronic_components",
        "chain_node": "pcb_and_display_touch_component",
        "node_role": "critical",
        "scores": (0.72, 0.62, 0.55, 0.60, 0.58, 0.32, 0.35, 0.45, 0.42, 0.35, 0.60, 0.62, 0.55, 0.50),
    },
    "002046": {
        "industry_chain": "precision_bearing_abrasive_materials",
        "chain_node": "high_end_bearing_and_superhard_material",
        "node_role": "critical",
        "scores": (0.70, 0.68, 0.58, 0.62, 0.55, 0.25, 0.30, 0.40, 0.35, 0.35, 0.58, 0.62, 0.58, 0.48),
    },
    "002119": {
        "industry_chain": "semiconductor_packaging_materials",
        "chain_node": "lead_frame_and_bonding_wire",
        "node_role": "bottleneck",
        "scores": (0.78, 0.72, 0.65, 0.66, 0.62, 0.28, 0.32, 0.42, 0.38, 0.45, 0.70, 0.66, 0.70, 0.58),
    },
    "002142": {
        "industry_chain": "yangtze_delta_finance",
        "chain_node": "sme_and_private_banking_credit",
        "node_role": "core",
        "scores": (0.78, 0.52, 0.58, 0.72, 0.48, 0.18, 0.24, 0.34, 0.28, 0.24, 0.36, 0.58, 0.36, 0.48),
    },
    "002585": {
        "industry_chain": "advanced_polymer_optical_film",
        "chain_node": "polyester_and_optical_film_substrate",
        "node_role": "critical",
        "scores": (0.68, 0.58, 0.58, 0.54, 0.50, 0.34, 0.38, 0.50, 0.45, 0.42, 0.58, 0.55, 0.52, 0.44),
    },
    "002812": {
        "industry_chain": "lithium_battery_separator",
        "chain_node": "wet_process_separator_material",
        "node_role": "bottleneck",
        "scores": (0.72, 0.68, 0.52, 0.64, 0.50, 0.30, 0.36, 0.45, 0.42, 0.45, 0.64, 0.62, 0.65, 0.55),
    },
    "003816": {
        "industry_chain": "nuclear_power_generation",
        "chain_node": "nuclear_power_operator",
        "node_role": "core",
        "scores": (0.88, 0.78, 0.56, 0.68, 0.54, 0.12, 0.20, 0.28, 0.24, 0.25, 0.60, 0.72, 0.45, 0.55),
    },
    "300221": {
        "industry_chain": "modified_polymer_materials",
        "chain_node": "modified_plastics_and_functional_material",
        "node_role": "alternative",
        "scores": (0.62, 0.48, 0.52, 0.46, 0.42, 0.35, 0.38, 0.55, 0.45, 0.42, 0.48, 0.44, 0.42, 0.36),
    },
    "300263": {
        "industry_chain": "advanced_materials_and_heat_exchange",
        "chain_node": "target_material_and_thermal_equipment",
        "node_role": "critical",
        "scores": (0.68, 0.62, 0.58, 0.58, 0.52, 0.28, 0.34, 0.46, 0.40, 0.42, 0.58, 0.58, 0.55, 0.48),
    },
    "300346": {
        "industry_chain": "semiconductor_materials",
        "chain_node": "photoresist_precursor_and_special_gas",
        "node_role": "bottleneck",
        "scores": (0.82, 0.78, 0.62, 0.70, 0.68, 0.22, 0.30, 0.42, 0.38, 0.45, 0.75, 0.70, 0.72, 0.62),
    },
    "300602": {
        "industry_chain": "ai_terminal_thermal_emi_materials",
        "chain_node": "thermal_management_and_emi_shielding",
        "node_role": "critical",
        "scores": (0.76, 0.64, 0.58, 0.62, 0.60, 0.26, 0.32, 0.42, 0.36, 0.42, 0.62, 0.66, 0.58, 0.52),
    },
    "300706": {
        "industry_chain": "pvd_coating_materials",
        "chain_node": "sputtering_target_material",
        "node_role": "bottleneck",
        "scores": (0.74, 0.72, 0.66, 0.62, 0.58, 0.30, 0.36, 0.44, 0.40, 0.48, 0.70, 0.62, 0.68, 0.55),
    },
    "300975": {
        "industry_chain": "electronic_component_distribution",
        "chain_node": "electronic_component_supply_chain_service",
        "node_role": "supporting",
        "scores": (0.66, 0.42, 0.55, 0.42, 0.46, 0.28, 0.32, 0.58, 0.42, 0.35, 0.44, 0.58, 0.38, 0.32),
    },
    "301389": {
        "industry_chain": "emi_shielding_functional_materials",
        "chain_node": "emi_shielding_and_precision_die_cut_material",
        "node_role": "critical",
        "scores": (0.74, 0.66, 0.58, 0.60, 0.56, 0.24, 0.30, 0.44, 0.38, 0.42, 0.62, 0.62, 0.58, 0.50),
    },
    "600895": {
        "industry_chain": "shanghai_semiconductor_innovation_park",
        "chain_node": "industrial_park_and_incubation_platform",
        "node_role": "supporting",
        "scores": (0.70, 0.46, 0.62, 0.50, 0.58, 0.20, 0.30, 0.46, 0.42, 0.38, 0.42, 0.60, 0.40, 0.36),
    },
    "600919": {
        "industry_chain": "jiangsu_regional_finance",
        "chain_node": "regional_bank_credit_and_wealth_management",
        "node_role": "core",
        "scores": (0.78, 0.52, 0.58, 0.72, 0.48, 0.18, 0.24, 0.34, 0.28, 0.24, 0.36, 0.58, 0.36, 0.48),
    },
    "601088": {
        "industry_chain": "coal_power_integrated_energy",
        "chain_node": "coal_resource_power_and_transport",
        "node_role": "core",
        "scores": (0.82, 0.74, 0.52, 0.70, 0.50, 0.10, 0.18, 0.30, 0.26, 0.28, 0.62, 0.72, 0.46, 0.58),
    },
    "601988": {
        "industry_chain": "systemically_important_bank",
        "chain_node": "large_state_owned_bank_credit_and_settlement",
        "node_role": "core",
        "scores": (0.86, 0.58, 0.52, 0.72, 0.42, 0.10, 0.18, 0.30, 0.24, 0.22, 0.35, 0.68, 0.35, 0.50),
    },
    "601998": {
        "industry_chain": "national_joint_stock_bank",
        "chain_node": "corporate_banking_and_interbank_finance",
        "node_role": "core",
        "scores": (0.80, 0.54, 0.54, 0.70, 0.44, 0.16, 0.22, 0.32, 0.26, 0.24, 0.35, 0.62, 0.35, 0.48),
    },
    "603288": {
        "industry_chain": "consumer_condiment_food",
        "chain_node": "condiment_brand_and_channel",
        "node_role": "core",
        "scores": (0.78, 0.54, 0.56, 0.78, 0.42, 0.08, 0.16, 0.34, 0.36, 0.28, 0.45, 0.74, 0.35, 0.68),
    },
    "603823": {
        "industry_chain": "organic_pigment_specialty_chemicals",
        "chain_node": "organic_pigment_manufacturing",
        "node_role": "critical",
        "scores": (0.66, 0.56, 0.60, 0.54, 0.48, 0.26, 0.32, 0.46, 0.40, 0.42, 0.54, 0.54, 0.50, 0.42),
    },
    "605358": {
        "industry_chain": "semiconductor_silicon_wafer_and_power_device",
        "chain_node": "silicon_wafer_and_discrete_device",
        "node_role": "bottleneck",
        "scores": (0.80, 0.74, 0.64, 0.66, 0.62, 0.24, 0.32, 0.42, 0.38, 0.48, 0.72, 0.66, 0.70, 0.58),
    },
    "605589": {
        "industry_chain": "synthetic_resin_biomass_materials",
        "chain_node": "phenolic_resin_and_biomass_chemical_material",
        "node_role": "critical",
        "scores": (0.70, 0.58, 0.56, 0.58, 0.50, 0.22, 0.30, 0.44, 0.38, 0.40, 0.54, 0.58, 0.50, 0.45),
    },
}


def _clean_code(value: Any) -> str:
    digits = "".join(ch for ch in str(value or "").strip() if ch.isdigit())
    return digits[-6:] if len(digits) >= 6 else digits.zfill(6) if digits else ""


def _clean_text(value: Any) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in {"", "nan", "none"} else text


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _mean(values: list[float | None]) -> float | None:
    cleaned = [value for value in values if value is not None]
    return sum(cleaned) / len(cleaned) if cleaned else None


def _ratio(value: float | None) -> float | None:
    if value is None:
        return None
    return value / 100.0 if abs(value) > 1.5 else value


def _round(value: float | None, digits: int = 6) -> float | None:
    return round(value, digits) if value is not None else None


def _read_csv(path: Path, **kwargs: Any) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _fetch_frame(
    *,
    raw_path: Path,
    force: bool,
    pause_seconds: float,
    call: Callable[[], pd.DataFrame],
) -> tuple[pd.DataFrame, str, str | None]:
    if raw_path.exists() and not force:
        return pd.read_csv(raw_path), "cache", None
    try:
        frame = call()
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(raw_path, index=False, encoding="utf-8-sig")
        if pause_seconds > 0:
            time.sleep(pause_seconds)
        return frame, "fetched", None
    except Exception as exc:  # Third-party interfaces fail with heterogeneous exceptions.
        return pd.DataFrame(), "failed", f"{type(exc).__name__}: {exc}"


def _annual_columns(frame: pd.DataFrame) -> list[str]:
    columns = [str(column) for column in frame.columns if str(column).isdigit() and str(column).endswith("1231")]
    return sorted(columns, reverse=True)


def _indicator_values(frame: pd.DataFrame, indicator: str, *, years: int) -> list[float | None]:
    if frame.empty or "指标" not in frame.columns:
        return []
    row = frame[frame["指标"].astype(str).eq(indicator)]
    if row.empty:
        row = frame[frame["指标"].astype(str).str.contains(indicator, regex=False, na=False)]
    if row.empty:
        return []
    cols = _annual_columns(frame)[:years]
    values = [_as_float(row.iloc[0].get(col)) for col in cols]
    return values


def _annual_report_rows(frame: pd.DataFrame, years: int) -> pd.DataFrame:
    if frame.empty or "报告日" not in frame.columns:
        return pd.DataFrame()
    data = frame.copy()
    data["报告日"] = data["报告日"].astype(str)
    data = data[data["报告日"].str.endswith("1231")]
    data = data.sort_values("报告日", ascending=False).drop_duplicates(subset=["报告日"], keep="first")
    return data.head(years)


def _compute_fcf(cash_flow: pd.DataFrame, abstract: pd.DataFrame) -> tuple[float | None, bool | None]:
    rows = _annual_report_rows(cash_flow, 5)
    if not rows.empty:
        ocf_values = []
        fcf_values = []
        for _, row in rows.iterrows():
            ocf = _as_float(row.get("经营活动产生的现金流量净额"))
            capex = _as_float(row.get("购建固定资产、无形资产和其他长期资产所支付的现金"))
            if ocf is not None:
                ocf_values.append(ocf)
                fcf_values.append(ocf - (capex or 0.0))
        if fcf_values:
            return sum(fcf_values), bool(ocf_values and ocf_values[0] > 0)
    ocf_values = _indicator_values(abstract, "经营现金流量净额", years=5)
    ocf_clean = [value for value in ocf_values if value is not None]
    return (sum(ocf_clean), bool(ocf_clean and ocf_clean[0] > 0)) if ocf_clean else (None, None)


def _compute_interest_coverage(income: pd.DataFrame, sector_type: str) -> float | None:
    if sector_type in {"bank", "financial"}:
        return None
    rows = _annual_report_rows(income, 5)
    coverages: list[float] = []
    for _, row in rows.iterrows():
        profit_total = _as_float(row.get("利润总额"))
        interest = _as_float(row.get("利息费用"))
        if interest is None or interest <= 0:
            interest = _as_float(row.get("利息支出"))
        if profit_total is not None and interest is not None and interest > 0:
            coverages.append((profit_total + interest) / interest)
    return _mean(coverages)


def _compute_share_dilution(share_change: pd.DataFrame) -> tuple[float | None, float | None]:
    if share_change.empty or "变动日期" not in share_change.columns or "总股本" not in share_change.columns:
        return None, None
    data = share_change.copy()
    data["变动日期"] = pd.to_datetime(data["变动日期"], errors="coerce")
    data["总股本"] = pd.to_numeric(data["总股本"], errors="coerce")
    data = data.dropna(subset=["变动日期", "总股本"]).sort_values("变动日期")
    if data.empty:
        return None, None
    latest = data.iloc[-1]
    latest_date = latest["变动日期"]
    latest_share = float(latest["总股本"])
    target_date = latest_date - pd.DateOffset(years=5)
    past = data[data["变动日期"] <= target_date]
    past_share = float(past.iloc[-1]["总股本"]) if not past.empty else float(data.iloc[0]["总股本"])
    dilution = (latest_share - past_share) / past_share if past_share > 0 else None
    first_date = data.iloc[0]["变动日期"]
    listed_years = (pd.Timestamp.today().normalize() - first_date).days / 365.25
    return dilution, listed_years


def _business_text(business: pd.DataFrame) -> tuple[str, str, str]:
    if business.empty:
        return "", "", ""
    row = business.iloc[0]
    main = _clean_text(row.get("主营业务"))
    product_type = _clean_text(row.get("产品类型"))
    product_name = _clean_text(row.get("产品名称"))
    return main, product_type, product_name


def _classify_sector(name: str, business_blob: str) -> str:
    text = f"{name} {business_blob}"
    if any(keyword in text for keyword in SECTOR_FINANCIAL_KEYWORDS):
        return "bank"
    if any(keyword in text for keyword in SECTOR_POWER_KEYWORDS):
        return "utility_energy"
    if any(keyword in text for keyword in SECTOR_CONSUMER_KEYWORDS):
        return "consumer"
    if any(keyword in text for keyword in SECTOR_TECH_KEYWORDS):
        return "technology_materials"
    return "industrial"


def _business_model(sector_type: str, business_blob: str) -> str:
    if sector_type == "bank":
        return "financial_credit_spread"
    if sector_type == "utility_energy":
        return "regulated_asset_operator"
    if "品牌" in business_blob or "食品" in business_blob or "调味" in business_blob:
        return "brand_channel"
    if any(keyword in business_blob for keyword in ("材料", "半导体", "电子", "光电", "封装")):
        return "critical_materials_component"
    if "物流" in business_blob or "港口" in business_blob:
        return "infrastructure_throughput"
    return "industrial_product_service"


def _chain_from_override(code: str, name: str, business_blob: str) -> dict[str, Any] | None:
    override = CHAIN_OVERRIDES.get(code)
    if not override:
        return None
    (
        demand_certainty,
        supply_constraint,
        attention_gap,
        value_capture,
        catalyst_strength,
        dilution_risk,
        financing_risk,
        substitution_risk,
        crowding_risk,
        valuation_risk,
        upstream_dependency,
        customer_validation,
        qualification_cycle,
        pricing_power,
    ) = override["scores"]
    note = f"AKShare/同花顺主营业务自动补齐；业务摘要：{business_blob[:180]}。产业链节点为规则初判，需人工复核。"
    return {
        "code": code,
        "name": name,
        "industry_chain": override["industry_chain"],
        "chain_node": override["chain_node"],
        "node_role": override["node_role"],
        "demand_certainty": demand_certainty,
        "supply_constraint": supply_constraint,
        "attention_gap": attention_gap,
        "value_capture": value_capture,
        "catalyst_strength": catalyst_strength,
        "dilution_risk": dilution_risk,
        "financing_risk": financing_risk,
        "substitution_risk": substitution_risk,
        "crowding_risk": crowding_risk,
        "valuation_risk": valuation_risk,
        "upstream_dependency": upstream_dependency,
        "customer_validation": customer_validation,
        "qualification_cycle": qualification_cycle,
        "pricing_power": pricing_power,
        "research_notes": note,
    }


def _chain_from_keywords(code: str, name: str, sector_type: str, business_blob: str) -> dict[str, Any]:
    if sector_type == "bank":
        chain = "regional_or_national_finance"
        node = "bank_credit_and_settlement"
        role = "core"
        scores = (0.76, 0.50, 0.54, 0.68, 0.42, 0.20, 0.25, 0.35, 0.30, 0.25, 0.35, 0.55, 0.35, 0.45)
    elif sector_type == "utility_energy":
        chain = "power_energy_infrastructure"
        node = "generation_or_resource_operator"
        role = "core"
        scores = (0.82, 0.70, 0.52, 0.64, 0.48, 0.15, 0.25, 0.30, 0.30, 0.28, 0.55, 0.65, 0.42, 0.52)
    elif sector_type == "consumer":
        chain = "consumer_food_brand"
        node = "brand_channel_and_food_manufacturing"
        role = "core"
        scores = (0.74, 0.46, 0.52, 0.68, 0.40, 0.14, 0.20, 0.38, 0.38, 0.30, 0.35, 0.66, 0.32, 0.58)
    elif sector_type == "technology_materials":
        chain = "advanced_electronic_materials"
        node = "critical_material_or_component"
        role = "critical"
        scores = (0.70, 0.62, 0.58, 0.58, 0.54, 0.28, 0.34, 0.45, 0.40, 0.42, 0.58, 0.58, 0.54, 0.48)
    else:
        chain = "industrial_supply_chain"
        node = "industrial_product_or_service"
        role = "supporting"
        scores = (0.62, 0.42, 0.50, 0.44, 0.38, 0.25, 0.30, 0.45, 0.35, 0.35, 0.42, 0.48, 0.38, 0.36)
    override = {"industry_chain": chain, "chain_node": node, "node_role": role, "scores": scores}
    CHAIN_OVERRIDES[code] = override
    return _chain_from_override(code, name, business_blob) or {}


def _qualitative_row(fundamental: dict[str, Any], chain_row: dict[str, Any], data_status: dict[str, str]) -> dict[str, Any]:
    gross_margin = _as_float(fundamental.get("gross_margin_5y_avg")) or 0.0
    roe = _as_float(fundamental.get("roe_10y_avg")) or 0.0
    ocf_quality = _as_float(fundamental.get("ocf_to_net_income_5y_avg")) or 0.0
    dilution = _as_float(fundamental.get("share_dilution_5y")) or 0.0
    fcf = _as_float(fundamental.get("fcf_5y_cumulative")) or 0.0
    role = str(chain_row.get("node_role") or "")

    circle = 4 if data_status.get("business") in {"fetched", "cache"} else 3
    moat = 2 + int(role in {"core", "critical"}) + int(role == "bottleneck") + int(gross_margin >= 0.30)
    management = 2 + int(ocf_quality >= 0.7) + int(ocf_quality >= 1.0) + int(dilution <= 0.10)
    safety = 2 + int(roe >= 0.08) + int(roe >= 0.15) + int(fcf > 0)
    thesis = 4 if chain_row.get("industry_chain") else 3
    red_flags = int(dilution > 0.25) + int(ocf_quality and ocf_quality < 0.50) + int(fcf < 0 and ocf_quality < 0.7)
    confidence = "B" if data_status.get("financial") in {"fetched", "cache"} and data_status.get("business") in {"fetched", "cache"} else "C"
    return {
        "code": fundamental["code"],
        "circle_of_competence_score": min(circle, 5),
        "moat_score": min(max(moat, 1), 5),
        "management_score": min(max(management, 1), 5),
        "safety_margin_score": min(max(safety, 1), 5),
        "thesis_clarity_score": min(thesis, 5),
        "red_flag_count": red_flags,
        "data_confidence": confidence,
        "qualitative_notes": "Auto-derived from public financial summary, share-change data, business scope, and industry-chain rule; review before promotion.",
    }


def _build_fundamental_row(
    *,
    code: str,
    name: str,
    abstract: pd.DataFrame,
    cash_flow: pd.DataFrame,
    income: pd.DataFrame,
    share_change: pd.DataFrame,
    business: pd.DataFrame,
) -> dict[str, Any]:
    main_business, product_type, product_name = _business_text(business)
    business_blob = "；".join(part for part in [main_business, product_type, product_name] if part)
    sector_type = _classify_sector(name, business_blob)
    fcf, recent_ocf_positive = _compute_fcf(cash_flow, abstract)
    share_dilution, listed_years = _compute_share_dilution(share_change)
    interest_coverage = _compute_interest_coverage(income, sector_type)
    net_margin_values = [_ratio(value) for value in _indicator_values(abstract, "销售净利率", years=10)]
    latest_two_net_margin = [value for value in net_margin_values[:2] if value is not None]
    research_notes = (
        "Auto-filled from AKShare interfaces: stock_financial_abstract, stock_financial_report_sina, "
        "stock_share_change_cninfo, stock_zyjs_ths. FCF uses OCF minus capex when available."
    )
    return {
        "code": code,
        "name": name,
        "sector_type": sector_type,
        "listed_years": _round(listed_years, 2),
        "roe_10y_avg": _round(_ratio(_mean(_indicator_values(abstract, "净资产收益率(ROE)", years=10)))),
        "fcf_5y_cumulative": _round(fcf, 2),
        "interest_coverage": _round(interest_coverage, 4),
        "gross_margin_5y_avg": _round(_ratio(_mean(_indicator_values(abstract, "毛利率", years=5)))),
        "ocf_to_net_income_5y_avg": _round(_mean(_indicator_values(abstract, "经营活动净现金/归属母公司的净利润", years=5))),
        "net_margin_10y_avg": _round(_mean(net_margin_values)),
        "share_dilution_5y": _round(share_dilution),
        "recent_ocf_positive": recent_ocf_positive,
        "net_margin_recovery": bool(len(latest_two_net_margin) == 2 and latest_two_net_margin[0] >= latest_two_net_margin[1]),
        "business_model_type": _business_model(sector_type, business_blob),
        "research_notes": research_notes,
    }


def run_enrichment(
    *,
    candidates_path: Path,
    existing_chain_map_path: Path | None,
    output_dir: Path,
    raw_dir: Path,
    force: bool,
    pause_seconds: float,
) -> dict[str, Any]:
    candidates = pd.read_csv(candidates_path, dtype={"code": str, "security_code": str})
    if "code" not in candidates.columns and "security_code" in candidates.columns:
        candidates["code"] = candidates["security_code"]
    if "name" not in candidates.columns and "security_name" in candidates.columns:
        candidates["name"] = candidates["security_name"]
    candidates["code"] = candidates["code"].map(_clean_code)
    candidates["name"] = candidates["name"].map(_clean_text)
    candidates = candidates[candidates["code"] != ""].drop_duplicates(subset=["code"], keep="first")

    existing_chain = _read_csv(existing_chain_map_path, dtype={"code": str}) if existing_chain_map_path else pd.DataFrame()
    if not existing_chain.empty:
        existing_chain["code"] = existing_chain["code"].map(_clean_code)
    existing_by_code = {str(row["code"]): row.to_dict() for _, row in existing_chain.iterrows()} if not existing_chain.empty else {}

    fundamental_rows: list[dict[str, Any]] = []
    qualitative_rows: list[dict[str, Any]] = []
    chain_rows_by_code: dict[str, dict[str, Any]] = {
        code: {column: row.get(column, "") for column in CHAIN_COLUMNS} for code, row in existing_by_code.items()
    }
    statuses: list[dict[str, Any]] = []

    for _, candidate in candidates.iterrows():
        code = str(candidate["code"])
        name = _clean_text(candidate.get("name")) or code
        code_raw_dir = raw_dir / code
        financial, financial_status, financial_error = _fetch_frame(
            raw_path=code_raw_dir / "stock_financial_abstract.csv",
            force=force,
            pause_seconds=pause_seconds,
            call=lambda code=code: ak.stock_financial_abstract(symbol=code),
        )
        cash_flow, cash_status, cash_error = _fetch_frame(
            raw_path=code_raw_dir / "stock_financial_cash_flow_sina.csv",
            force=force,
            pause_seconds=pause_seconds,
            call=lambda code=code: ak.stock_financial_report_sina(stock=code, symbol="现金流量表"),
        )
        income, income_status, income_error = _fetch_frame(
            raw_path=code_raw_dir / "stock_financial_income_sina.csv",
            force=force,
            pause_seconds=pause_seconds,
            call=lambda code=code: ak.stock_financial_report_sina(stock=code, symbol="利润表"),
        )
        share_change, share_status, share_error = _fetch_frame(
            raw_path=code_raw_dir / "stock_share_change_cninfo.csv",
            force=force,
            pause_seconds=pause_seconds,
            call=lambda code=code: ak.stock_share_change_cninfo(symbol=code),
        )
        business, business_status, business_error = _fetch_frame(
            raw_path=code_raw_dir / "stock_zyjs_ths.csv",
            force=force,
            pause_seconds=pause_seconds,
            call=lambda code=code: ak.stock_zyjs_ths(symbol=code),
        )

        fundamental = _build_fundamental_row(
            code=code,
            name=name,
            abstract=financial,
            cash_flow=cash_flow,
            income=income,
            share_change=share_change,
            business=business,
        )
        fundamental_rows.append(fundamental)

        if code in existing_by_code:
            chain_row = {column: existing_by_code[code].get(column, "") for column in CHAIN_COLUMNS}
        else:
            main_business, product_type, product_name = _business_text(business)
            business_blob = "；".join(part for part in [main_business, product_type, product_name] if part)
            chain_row = _chain_from_override(code, name, business_blob)
            if chain_row is None:
                chain_row = _chain_from_keywords(code, name, str(fundamental.get("sector_type") or ""), business_blob)
        chain_rows_by_code[code] = chain_row

        status = {
            "code": code,
            "name": name,
            "financial": financial_status,
            "cash_flow": cash_status,
            "income": income_status,
            "share_change": share_status,
            "business": business_status,
            "financial_error": financial_error,
            "cash_flow_error": cash_error,
            "income_error": income_error,
            "share_change_error": share_error,
            "business_error": business_error,
        }
        qualitative_rows.append(_qualitative_row(fundamental, chain_row, status))
        statuses.append(status)
        print(
            f"{code} {name}: financial={financial_status}, cash={cash_status}, "
            f"income={income_status}, shares={share_status}, business={business_status}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    fundamentals_path = output_dir / "berkshire_fundamentals_latest.csv"
    qualitative_path = output_dir / "berkshire_qualitative_latest.csv"
    chain_path = output_dir / "industry_chain_map_enriched_latest.csv"
    status_path = output_dir / "candidate_online_enrichment_status_latest.csv"
    manifest_path = output_dir / "candidate_online_enrichment_manifest_latest.json"

    fundamentals = pd.DataFrame(fundamental_rows, columns=FUNDAMENTAL_COLUMNS)
    qualitative = pd.DataFrame(qualitative_rows, columns=QUALITATIVE_COLUMNS)
    chain = pd.DataFrame(list(chain_rows_by_code.values()), columns=CHAIN_COLUMNS)
    status_frame = pd.DataFrame(statuses)
    _write_csv(fundamentals, fundamentals_path)
    _write_csv(qualitative, qualitative_path)
    _write_csv(chain, chain_path)
    _write_csv(status_frame, status_path)

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "research_only": True,
        "broker_action": "none",
        "candidates_path": str(candidates_path),
        "existing_chain_map_path": None if existing_chain_map_path is None else str(existing_chain_map_path),
        "output_dir": str(output_dir),
        "raw_dir": str(raw_dir),
        "candidate_count": int(len(candidates)),
        "fundamental_count": int(len(fundamentals)),
        "qualitative_count": int(len(qualitative)),
        "industry_chain_count": int(len(chain)),
        "new_chain_rows": int(sum(1 for code in candidates["code"] if code not in existing_by_code)),
        "sources": [
            {
                "name": "AKShare stock_financial_abstract",
                "upstream": "Eastmoney financial summary",
                "url": "https://akshare.akfamily.xyz/data/stock/stock.html",
            },
            {
                "name": "AKShare stock_financial_report_sina",
                "upstream": "Sina financial statements",
                "url": "https://akshare.akfamily.xyz/data/stock/stock.html",
            },
            {
                "name": "AKShare stock_share_change_cninfo",
                "upstream": "CNINFO share change",
                "url": "https://akshare.akfamily.xyz/data/stock/stock.html",
            },
            {
                "name": "AKShare stock_zyjs_ths",
                "upstream": "Tonghuashun business scope",
                "url": "https://akshare.akfamily.xyz/data/stock/stock.html",
            },
        ],
        "method_notes": [
            "FCF is approximated as operating cash flow minus capex when cash-flow statement fields are available.",
            "Industry-chain nodes for missing stocks are rule-based initial research labels from public business descriptions; they require review before promotion.",
            "Qualitative scores are model-derived helper fields, not independent analyst ratings.",
        ],
        "outputs": {
            "fundamentals_path": str(fundamentals_path),
            "qualitative_path": str(qualitative_path),
            "chain_path": str(chain_path),
            "status_path": str(status_path),
            "manifest_path": str(manifest_path),
        },
        "status_counts": {
            column: status_frame[column].value_counts(dropna=False).to_dict()
            for column in ["financial", "cash_flow", "income", "share_change", "business"]
            if column in status_frame.columns
        },
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Enrich stock candidates with public research data.")
    parser.add_argument("--candidates-path", default="outputs/research/paper_account_latest/stock_target_review.csv")
    parser.add_argument("--existing-chain-map-path", default="data/research/industry_chain_map.csv")
    parser.add_argument("--output-dir", default="data/research")
    parser.add_argument("--raw-dir", default="data/research/online_enrichment/raw")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--pause-seconds", type=float, default=0.35)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    existing_chain_map_path = Path(args.existing_chain_map_path) if args.existing_chain_map_path else None
    manifest = run_enrichment(
        candidates_path=Path(args.candidates_path),
        existing_chain_map_path=existing_chain_map_path,
        output_dir=Path(args.output_dir),
        raw_dir=Path(args.raw_dir),
        force=args.force,
        pause_seconds=args.pause_seconds,
    )
    print(json.dumps(manifest["outputs"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
