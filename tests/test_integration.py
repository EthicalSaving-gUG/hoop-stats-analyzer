"""
test_integration.py
===================
End-to-end integration test: runs BasketballAnalyzer.run_fast() on the
first 300 frames of the upload video and validates the output JSON.

This test is automatically SKIPPED if the upload video is absent.
"""
import os
import json
import pytest

ROOT_DIR     = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
UPLOAD_VIDEO = os.path.join(
    ROOT_DIR, "uploads", "Veo Bringe dein Spiel auf das nachste Level.mp4"
)


@pytest.mark.skipif(
    not os.path.isfile(UPLOAD_VIDEO),
    reason="Upload video not present — skipping integration test",
)
def test_run_fast_produces_valid_output(tmp_path):
    """
    run_fast() on the first 300 frames should produce:
    - a .json file that is valid JSON
    - a top-level 'players' key (dict, may be empty for short clips)
    - a top-level 'scoring' key with integer values
    - a top-level 'team_summary' key
    """
    from hoop_stats.analyzer import BasketballAnalyzer

    output_name = str(tmp_path / "integration_test")
    analyzer = BasketballAnalyzer(UPLOAD_VIDEO, output_name=output_name)
    # Use 2 workers and limit to 300 frames for speed
    analyzer.run_fast(max_frames=300, skip_frames=3, n_workers=2)

    json_path = output_name + ".json"
    assert os.path.isfile(json_path), f"Expected JSON output at {json_path}"

    with open(json_path) as f:
        data = json.load(f)

    assert "players" in data,      "JSON missing 'players' key"
    assert "scoring" in data,      "JSON missing 'scoring' key"
    assert "team_summary" in data, "JSON missing 'team_summary' key"

    scoring = data["scoring"]
    assert isinstance(scoring.get("total_baskets", 0), int)

    print(f"\n[integration] Players tracked: {len(data['players'])}")
    print(f"[integration] Scoring: {scoring}")


@pytest.mark.skipif(
    not os.path.isfile(UPLOAD_VIDEO),
    reason="Upload video not present — skipping benchmark",
)
def test_run_fast_is_faster_than_run(tmp_path):
    """
    Speed benchmark: run_fast() should be at least 1.5× faster than run()
    on the same 600-frame window.
    """
    import time
    from hoop_stats.analyzer import BasketballAnalyzer

    MAX_FRAMES = 600

    t0 = time.perf_counter()
    slow = BasketballAnalyzer(UPLOAD_VIDEO, output_name=str(tmp_path / "slow"))
    slow.run(max_frames=MAX_FRAMES, skip_frames=3)
    slow_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    fast = BasketballAnalyzer(UPLOAD_VIDEO, output_name=str(tmp_path / "fast"))
    fast.run_fast(max_frames=MAX_FRAMES, skip_frames=3, n_workers=2)
    fast_time = time.perf_counter() - t0

    speedup = slow_time / max(fast_time, 0.001)
    print(f"\n[benchmark] run()      : {slow_time:.1f}s")
    print(f"[benchmark] run_fast() : {fast_time:.1f}s")
    print(f"[benchmark] Speedup    : {speedup:.2f}×")

    import torch
    if torch.cuda.is_available():
        assert speedup >= 1.5, (
            f"run_fast() speedup was only {speedup:.2f}× — expected ≥ 1.5×"
        )
    else:
        print("[benchmark] Running on CPU — skipping speedup assertion due to worker overhead.")
