import sys
import argparse
from hoop_stats.analyzer import BasketballAnalyzer

def main():
    parser = argparse.ArgumentParser(description="Hoop Stats Analyzer")
    parser.add_argument("video", help="Path to video file or RTSP stream URL")
    parser.add_argument("--output", default="output_stats", help="Base name for output files")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Hard limit on frames to process (useful for quick tests)")
    parser.add_argument("--skip-frames", type=int, default=3,
                        help="Process every Nth frame (default=3 for 3x speed, use 1 for full accuracy)")
    parser.add_argument("--calibrate", action="store_true",
                        help="Use default court calibration (28m x 15m FIBA court)")

    args = parser.parse_args()

    calibration = None
    if args.calibrate:
        print("Using default FIBA court calibration (28m x 15m)...")
        calibration = {
            "video": [[100, 100], [1100, 100], [100, 900], [1100, 900]],
            "real":  [[0, 0], [28, 0], [0, 15], [28, 15]]
        }

    analyzer = BasketballAnalyzer(args.video, args.output)
    analyzer.run(
        court_calibration_points=calibration,
        max_frames=args.max_frames,
        skip_frames=args.skip_frames
    )

if __name__ == "__main__":
    main()
