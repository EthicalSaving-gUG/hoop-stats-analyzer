import pytest
import sys
import os

from hoop_stats.shot_tracker import ShotTracker

# Instead of importing TestShotTracker directly which pytest might intercept,
# we can rely on `pytest tests/` which already runs these successfully.
print("Pytest handles these tests, skipped manual runner.")
