"""Test import time of core modules to avoid regression."""

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
    "pymatviz": 4085,
    "pymatviz.coordination": 4135,
    "pymatviz.histogram": 4110,
    "pymatviz.phonons": 4109,
    "pymatviz.powerups": 4066,
    "pymatviz.ptable": 4092,
    "pymatviz.rainclouds": 4098,
    "pymatviz.rdf": 4144,
    "pymatviz.classify": 4126,
    "pymatviz.sankey": 4135,
    "pymatviz.scatter": 4087,
    "pymatviz.structure": 4105,
    "pymatviz.sunburst": 4133,
    "pymatviz.uncertainty": 4179,
    "pymatviz.xrd": 4156,
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
        repeats (int): Number of runs to average.

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
        grace_percent (float): Maximum allowed percentage increase in import time
            before a warning is raised.
        hard_percent (float): Maximum allowed percentage increase in import time
            before the test fails.
    """
    for module_name, ref_time in REF_IMPORT_TIME.items():
        current_time = measure_import_time(module_name)

        # Calculate grace and hard thresholds
        grace_threshold = ref_time * (1 + grace_percent)
        hard_threshold = ref_time * (1 + hard_percent)

        if current_time > grace_threshold:
            if current_time > hard_threshold:
                pytest.fail(
                    f"{module_name} import too slow! took {current_time:.0f} ms, "
                    f"{hard_threshold=:.0f} ms"
                )
            else:
                warnings.warn(
                    f"{module_name} import slightly slower: took {current_time:.0f} "
                    f"ms, {grace_threshold=:.0f} ms",
                    stacklevel=2,
                )
