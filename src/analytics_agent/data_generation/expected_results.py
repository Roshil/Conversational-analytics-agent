"""Independent expected (gold) results.

The implementation lives in ``analytics_agent.expected_results``. This module re-exports it so the data
generator and the evaluation always use the same, single set of gold cases (a second copy here had
fallen behind and was missing the hard cases G21-G30).
"""
from analytics_agent.expected_results import *  # noqa: F401,F403
from analytics_agent.expected_results import build_expected_results  # noqa: F401
