"""
Test import time of core modules to avoid regression.
"""

# ruff: noqa: T201 (check for print statement)

from __future__ import annotations

import os
import subprocess
import time

import pytest


GEN_REF_TIME = False  # switch for generating reference time

# Last update: 2024-10-23
REF_IMPORT_TIME: dict[str, float] = {
    "pymatviz": 4085.73,
    "pymatviz.coordination": 4135.77,
    "pymatviz.cumulative": 4108.06,
    "pymatviz.histogram": 4110.41,
    "pymatviz.phonons": 4109.97,
    "pymatviz.powerups": 4066.31,
    "pymatviz.ptable": 4092.35,
    "pymatviz.rainclouds": 4098.33,
    "pymatviz.rdf": 4144.26,
    "pymatviz.relevance": 4126.54,
    "pymatviz.sankey": 4135.17,
    "pymatviz.scatter": 4087.62,
    "pymatviz.structure_viz": 4105.33,
    "pymatviz.sunburst": 4133.78,
    "pymatviz.uncertainty": 4179.99,
    "pymatviz.xrd": 4156.52,
}


@pytest.mark.skipif(
    not GEN_REF_TIME, reason="Set GEN_REF_TIME to generate reference import time."
)
def test_get_ref_import_time() -> None:
    """A dummy test that would always fail, used to generate copyable reference time."""
    import_times = {
        module_name: measure_import_time_in_ms(module_name)
        for module_name in REF_IMPORT_TIME
    }

    # Print out the import times in a copyable format
    print("\nCopyable import time dictionary:")
    print("{")
    for module_name, import_time in import_times.items():
        print(f'    "{module_name}": {import_time:.2f},')
    print("}")

    pytest.fail("Generate reference import times.")


def measure_import_time_in_ms(module_name: str, count: int = 3) -> float:
    """Measure import time of a module in milliseconds across several runs.

    Args:
        module_name (str): name of the module to test.
        count (int): Number of runs to average.

    Returns:
        float: import time in milliseconds.
    """
    total_time = 0.0

    for _ in range(count):
        start_time = time.perf_counter()
        subprocess.run(["python", "-c", f"import {module_name}"], check=True)  # noqa: S603, S607
        total_time += time.perf_counter() - start_time

    return (total_time / count) * 1000


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
        current_time = measure_import_time_in_ms(module_name)

        # Calculate grace and hard thresholds
        grace_threshold = ref_time * (1 + grace_percent)
        hard_threshold = ref_time * (1 + hard_percent)

        if current_time > grace_threshold:
            if current_time > hard_threshold:
                pytest.fail(f"{module_name} import too slow! {hard_threshold=:.2f} ms")
            else:
                pytest.warns(
                    UserWarning,
                    f"{module_name} import slightly slower: {grace_threshold=:.2f} ms",
                )
