import argparse
from ast import Import
import json
import sys
from pathlib import Path
from multiprocessing import Process
from tracker import DBSCAN_Tracker

# def _run_tracker(raw_path: str) -> None:
#     Tracker(raw_path)

# def main() -> None:
#     parser = argparse.ArgumentParser(description="Run two Tracker instances in parallel.")
#     parser.add_argument("raw_path_a", type=str, help="First raw file path")
#     parser.add_argument("raw_path_b", type=str, help="Second raw file path")
#     args = parser.parse_args()

#     paths = [Path(args.raw_path_a), Path(args.raw_path_b)]
#     for p in paths:
#         if not p.exists():
#             parser.error(f"File not found: {p}")

#     p1 = Process(target=_run_tracker, args=(str(paths[0]),))
#     p2 = Process(target=_run_tracker, args=(str(paths[1]),))

#     p1.start()
#     p2.start()
#     p1.join()
#     p2.join()

#     if p1.exitcode != 0 or p2.exitcode != 0:
#         sys.exit(1)

DELTA_T = 100000  #microseconds
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Tracker on a raw file.")
    parser.add_argument("rawfile", type=str, help="Path to the raw file")
    args = parser.parse_args()

    raw_path = Path(args.rawfile)
    if not raw_path.exists():
        parser.error(f"File not found: {raw_path}")

    tracker = DBSCAN_Tracker()
    tracker.run_tracker(str(raw_path), hierarchical=True, plot_clusters=True)