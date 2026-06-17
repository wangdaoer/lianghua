"""Compatibility package for running from the project root without install."""

from pathlib import Path

_SRC_PACKAGE = Path(__file__).resolve().parent.parent / "src" / "quant_etf_lab"
if _SRC_PACKAGE.is_dir():
    __path__.append(str(_SRC_PACKAGE))

