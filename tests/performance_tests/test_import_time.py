"""
Test import time of core modules to avoid regression.
"""

# ruff: noqa: T201 (check for print statement)

from __future__ import annotations

import subprocess
import time

import pytest


# TODO: add more test modules
REF_IMPORT_TIME: dict[str, float] = {
    "pymatviz": None,
    "pymatviz.ptable": None,
    "pymatviz.io": None,
    "pymatviz.scatter": None,
    "pymatviz.phonons": None,
}


# @pytest.mark.skip(reason="Unskip to generate reference import time.")
def test_get_ref_import_time() -> None:
    """A dummy test that would always fail, used to generate copyable reference time."""
    # Measure import time for each module
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

    pytest.fail("Generated reference import times.")


def measure_import_time_in_ms(module_name: str, count: int = 10) -> float:
    """Measure import time of a module in milliseconds across several runs.

    Args:
        module_name (str): name of the module to test.
        count (int): Number of runs to average.

    Returns:
        float: import time in milliseconds.
    """
    total_time = 0.0

    for _ in range(count):
        start_time = time.time()
        subprocess.run(["python", "-c", f"import {module_name}"], check=True)  # noqa: S603, S607
        total_time += time.time() - start_time

    return (total_time / count) * 1000


def test_import_time(grace_percent: float = 0.20, hard_percent: float = 0.50) -> None:
    """Test the import time of core modules to avoid regression in performance.

    Args:
        grace_percentage (float): Maximum allowed percentage increase in import time
            before a warning is raised.
        hard_percentage (float): Maximum allowed percentage increase in import time
            before the test fails.
    """
    for module_name, ref_time in REF_IMPORT_TIME.items():
        if ref_time is None:
            pytest.skip(f"No reference import time for {module_name}")

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
