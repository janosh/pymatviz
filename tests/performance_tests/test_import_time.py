"""
Test import time of core modules to avoid regression.
"""

# ruff: noqa: T201 (check for print statement)

from __future__ import annotations

import os
import subprocess
import time
import warnings

import pytest


GEN_REF_TIME = False  # switch for generating reference time

# Last update: 2024-10-23
REF_IMPORT_TIME: dict[str, float] = {
    "pymatviz": 2084.25,
    "pymatviz.coordination": 2342.41,
    "pymatviz.cumulative": 2299.73,
    "pymatviz.histogram": 2443.11,
    "pymatviz.phonons": 2235.57,
    "pymatviz.powerups": 2172.71,
    "pymatviz.ptable": 2286.77,
    "pymatviz.rainclouds": 2702.03,
    "pymatviz.rdf": 2331.98,
    "pymatviz.relevance": 2256.29,
    "pymatviz.sankey": 2313.12,
    "pymatviz.scatter": 2312.48,
    "pymatviz.structure_viz": 2330.39,
    "pymatviz.sunburst": 2395.04,
    "pymatviz.uncertainty": 2317.87,
    "pymatviz.xrd": 2242.09,
}


@pytest.mark.skipif(
    not GEN_REF_TIME, reason="Set GEN_REF_TIME to generate reference import time."
)
def test_get_ref_import_time() -> None:
    """A dummy test that would always fail, used to generate copyable reference time."""
    import_times = {
        module_name: round(measure_import_time(module_name), 2)
        for module_name in REF_IMPORT_TIME
    }

    # Print out the import times in a copyable format
    print("\nCopyable import time dictionary:")
    print(import_times)

    pytest.fail("Generate reference import times.")


def measure_import_time(module_name: str, repeats: int = 3) -> float:
    """Measure import time of a module in milliseconds across several runs.

    Args:
        module_name (str): name of the module to test.
        count (int): Number of runs to average.

    Returns:
        float: import time in milliseconds.
    """
    total_time = 0.0

    for _ in range(repeats):
        start_time = time.perf_counter()
        subprocess.run(["python", "-c", f"import {module_name}"], check=True)  # noqa: S603, S607
        total_time += time.perf_counter() - start_time

    return total_time / repeats * 1000


@pytest.mark.skipif(
    os.getenv("GITHUB_REF") != "refs/heads/main", reason="Only run on the main branch"
)
@pytest.mark.skipif(GEN_REF_TIME, reason="Generating reference import time.")
def test_import_time(grace_percent: float = 0.20, hard_percent: float = 0.50) -> None:
    """Test the import time of core modules to avoid regression in performance.

    Args:
        grace_percentage (float): Maximum allowed percentage increase in import time
            before a warning is raised.
        hard_percentage (float): Maximum allowed percentage increase in import time
            before the test fails.
    """
    for module_name, ref_time in REF_IMPORT_TIME.items():
        current_time = measure_import_time(module_name)

        # Calculate grace and hard thresholds
        grace_threshold = ref_time * (1 + grace_percent)
        hard_threshold = ref_time * (1 + hard_percent)

        if current_time > grace_threshold:
            if current_time > hard_threshold:
                pytest.fail(f"{module_name} import too slow! {hard_threshold=:.2f} ms")
            else:
                warnings.warn(
                    f"{module_name} import slightly slower: {grace_threshold=:.2f} ms",
                    stacklevel=2,
                )
