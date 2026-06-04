#!/usr/bin/env python3
"""Generate an ArUco marker PNG for Stage 1 (DICT_4X4_50, ID 0, 200×200 px)."""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Generate ArUco marker PNG (requires opencv-python)"
    )
    parser.add_argument("--output",    default="aruco_marker.png")
    parser.add_argument("--marker_id", type=int, default=0,
                        help="Marker ID (0–49 for DICT_4X4_50)")
    parser.add_argument("--size_px",   type=int, default=200,
                        help="Output image side length in pixels")
    args = parser.parse_args()

    try:
        import cv2
    except ImportError:
        print("cv2 not found — run:  pip install opencv-python", file=sys.stderr)
        sys.exit(1)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    # generateImageMarker is available in OpenCV 4.7+;
    # older versions use drawMarker.
    if hasattr(cv2.aruco, "generateImageMarker"):
        img = cv2.aruco.generateImageMarker(aruco_dict, args.marker_id, args.size_px)
    else:
        img = cv2.aruco.drawMarker(aruco_dict, args.marker_id, args.size_px)

    cv2.imwrite(args.output, img)
    print(f"Saved DICT_4X4_50 marker {args.marker_id} "
          f"({args.size_px}×{args.size_px} px) → {args.output}")


if __name__ == "__main__":
    main()
