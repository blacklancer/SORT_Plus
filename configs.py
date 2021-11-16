from __future__ import print_function

import matplotlib
matplotlib.use('TkAgg')
import argparse


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display',
                        dest='display',
                        help='Display online tracker output (slow) [False]',
                        action='store_true')
    parser.add_argument("--seq_path",
                        help="Path to detections.",
                        type=str,
                        default='data')
    parser.add_argument("--phase",
                        help="Subdirectory in seq_path.",
                        type=str,
                        default='train')
    parser.add_argument("--max_age",
                        help="Maximum number of frames to keep alive a track without associated detections.",
                        type=int, default=1)
    parser.add_argument("--min_hits",
                        help="Minimum number of associated detections before track is initialised.",
                        type=int, default=3)
    parser.add_argument("--iou_threshold",
                        help="Minimum IOU for match.",
                        type=float,
                        default=0.3)
    args = parser.parse_args()
    return args