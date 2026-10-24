"""Policy-specific table checkers, chosen by the test set's ``table_checker`` field."""
from typing import Callable, Dict, Optional


def get_table_checker(name: Optional[str]) -> Optional[Callable]:
    """``check(answer, cells, question) -> {"expected", "correct", "wrong", "missing", "year_not_named"}``"""
    if not name:
        return None
    if name == "optima_secure":
        from .optima_secure import table_check
        return table_check
    raise KeyError(f"unknown table checker {name!r}")
