#!/usr/bin/env python3
"""
@file data_prep.py
@brief Offline data preparation: download CIFAR-10 and extract video frames.

Phase 1: Download CIFAR-10 binary dataset (~163 MB).
Phase 2: Extract frames from a surveillance video, resize to 32x32x3,
         save as raw float32 binary for the C inference pipeline.

Usage:
  python data_prep.py [--video VIDEO_PATH] [--fps 2] [--max-frames 7200]
"""

import os
import sys
import argparse
import array
import subprocess
import tarfile
import shutil
import urllib.request
import struct
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset")
VIDEO_DIR = os.path.join(SCRIPT_DIR, "video_frames")

CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
CIFAR10_FILES = [
    "data_batch_1.bin", "data_batch_2.bin", "data_batch_3.bin",
    "data_batch_4.bin", "data_batch_5.bin", "test_batch.bin",
    "batches.meta.txt"
]

FRAME_WIDTH = 32
FRAME_HEIGHT = 32
FRAME_CHANNELS = 3
FRAME_PIXELS = FRAME_WIDTH * FRAME_HEIGHT * FRAME_CHANNELS  # 3072


def download_cifar10():
    """Download and extract CIFAR-10 binary dataset."""
    os.makedirs(DATASET_DIR, exist_ok=True)

    # Check if already downloaded
    all_exist = all(
        os.path.exists(os.path.join(DATASET_DIR, f)) for f in CIFAR10_FILES
    )
    if all_exist:
        print("[CIFAR-10] All files already present, skipping download.")
        return True

    print("[CIFAR-10] Downloading from {} ...".format(CIFAR10_URL))
    tar_path = os.path.join(DATASET_DIR, "cifar-10-binary.tar.gz")

    try:
        urllib.request.urlretrieve(CIFAR10_URL, tar_path)
    except Exception as e:
        print("[CIFAR-10] Download failed: {}".format(e))
        print("[CIFAR-10] Please download manually from:")
        print("  https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz")
        print("  and extract to: {}".format(DATASET_DIR))
        return False

    print("[CIFAR-10] Extracting ...")
    with tempfile.TemporaryDirectory() as tmpdir:
        with tarfile.open(tar_path, "r:gz") as tar:
            tar.extractall(tmpdir)
        extracted = os.path.join(tmpdir, "cifar-10-batches-bin")
        for fname in os.listdir(extracted):
            src = os.path.join(extracted, fname)
            dst = os.path.join(DATASET_DIR, fname)
            shutil.copy2(src, dst)

    os.remove(tar_path)
    print("[CIFAR-10] Done. {} files in {}.".format(len(CIFAR10_FILES), DATASET_DIR))
    return True


def check_ffmpeg():
    """Check if ffmpeg is available."""
    try:
        subprocess.run(["ffmpeg", "-version"], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def extract_video_frames(video_path, fps=2, max_frames=7200):
    """Extract frames from video, resize to 32x32, save as raw float32."""
    if not check_ffmpeg():
        print("[Video] ffmpeg not found. Install ffmpeg or place it in PATH.")
        print("[Video] Alternatively, pre-extract frames manually into:")
        print("  {}".format(VIDEO_DIR))
        print("[Video] Expected: 32x32x3 raw float32 frames in video_frames.dat")
        print("[Video] With video_meta.txt describing frame count and format.")
        return False

    if not os.path.exists(video_path):
        print("[Video] Video file not found: {}".format(video_path))
        return False

    os.makedirs(VIDEO_DIR, exist_ok=True)

    print("[Video] Extracting frames from: {}".format(video_path))
    print("[Video] Parameters: fps={}, max_frames={}, resize=32x32".format(
        fps, max_frames))

    # ffmpeg command: extract frames as raw rgb24, pipe to python for float32 conversion
    ffmpeg_cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", "fps={},scale={}:{}".format(fps, FRAME_WIDTH, FRAME_HEIGHT),
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-vframes", str(max_frames),
        "pipe:1"
    ]

    output_path = os.path.join(VIDEO_DIR, "video_frames.dat")
    meta_path = os.path.join(VIDEO_DIR, "video_meta.txt")

    frame_count = 0

    try:
        proc = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE)
    except Exception as e:
        print("[Video] Failed to start ffmpeg: {}".format(e))
        return False

    with open(output_path, "wb") as f:
        while True:
            raw_bytes = proc.stdout.read(FRAME_PIXELS)
            if len(raw_bytes) < FRAME_PIXELS:
                break

            # Convert uint8 [0,255] to float32 [0,1]
            # Use array.array with generator to avoid large intermediate list
            arr = array.array("f", (b / 255.0 for b in raw_bytes))
            f.write(arr.tobytes())
            frame_count += 1

            if frame_count % 500 == 0:
                print("  Extracted {} frames ...".format(frame_count))

    proc.wait()
    stderr_output = proc.stderr.read()
    # ffmpeg prints stats to stderr, not necessarily errors

    print("[Video] Extracted {} frames total.".format(frame_count))

    if frame_count == 0:
        print("[Video] No frames extracted. Check the video file and ffmpeg output:")
        print(stderr_output.decode(errors="replace")[:2000])
        return False

    # Write metadata
    with open(meta_path, "w") as mf:
        mf.write("frame_count {}\n".format(frame_count))
        mf.write("fps {}\n".format(fps))
        mf.write("width {}\n".format(FRAME_WIDTH))
        mf.write("height {}\n".format(FRAME_HEIGHT))
        mf.write("channels {}\n".format(FRAME_CHANNELS))

    data_size_mb = frame_count * FRAME_PIXELS * 4 / (1024 * 1024)
    print("[Video] Saved to: {}".format(output_path))
    print("[Video] Data size: {:.1f} MB".format(data_size_mb))
    print("[Video] Metadata: {}".format(meta_path))
    return True


def create_synthetic_frames(num_frames=7200):
    """Create synthetic video frames for demo purposes when no real video is available."""
    import random
    random.seed(42)

    os.makedirs(VIDEO_DIR, exist_ok=True)

    output_path = os.path.join(VIDEO_DIR, "video_frames.dat")
    meta_path = os.path.join(VIDEO_DIR, "video_meta.txt")

    print("[Video] No video file provided. Generating synthetic frames for demo.")
    print("[Video] Creating {} frames of synthetic surveillance-style noise ...".format(num_frames))

    with open(output_path, "wb") as f:
        for i in range(num_frames):
            # Generate frames with some structure to make motion detection meaningful
            frame = []
            # Background: dark with slight noise
            if i % 100 < 30:
                # "Motion block" - slightly brighter region simulating movement
                for y in range(FRAME_HEIGHT):
                    for x in range(FRAME_WIDTH):
                        if 8 <= x < 24 and 8 <= y < 24:
                            # Moving object region
                            phase = (i % 30) * 0.2
                            r = min(1.0, max(0.0, 0.3 + 0.2 * (
                                (x - 16 + 4 * (i % 30 - 15)) ** 2 +
                                (y - 16 + 4 * ((i // 30) % 15 - 7)) ** 2
                            ) ** 0.5 * 0.02))
                            g = r * 0.8
                            b = r * 0.6
                        else:
                            r = random.uniform(0.0, 0.05)
                            g = random.uniform(0.0, 0.05)
                            b = random.uniform(0.0, 0.05)
                        frame.extend([r, g, b])
            else:
                # Static frame: just dark background noise
                frame = [random.uniform(0.0, 0.05) for _ in range(FRAME_PIXELS)]

            f.write(struct.pack("{}f".format(FRAME_PIXELS), *frame))

            if (i + 1) % 1000 == 0:
                print("  Generated {} frames ...".format(i + 1))

    with open(meta_path, "w") as mf:
        mf.write("frame_count {}\n".format(num_frames))
        mf.write("fps 2\n")
        mf.write("width {}\n".format(FRAME_WIDTH))
        mf.write("height {}\n".format(FRAME_HEIGHT))
        mf.write("channels {}\n".format(FRAME_CHANNELS))

    print("[Video] Synthetic frames saved to: {}".format(output_path))
    print("[Video] NOTE: These are synthetic, for pipeline testing only.")
    print("[Video] For real surveillance data, provide a video file with --video.")
    print("[Video] Recommended: MEVA dataset (CC-BY-4.0, 328+ hours CCTV)")
    print("[Video]   Download from: s3://mevadata-public-01/drops-123-r13")
    return True


def parse_video_meta():
    """Read video_meta.txt and return frame count and metadata."""
    meta_path = os.path.join(VIDEO_DIR, "video_meta.txt")
    if not os.path.exists(meta_path):
        return None
    meta = {}
    with open(meta_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                meta[parts[0]] = int(parts[1])
    return meta


def main():
    parser = argparse.ArgumentParser(
        description="Edge Video Preprocessing Demo - Data Preparation"
    )
    parser.add_argument("--video", type=str, default=None,
                        help="Path to surveillance video file")
    parser.add_argument("--fps", type=int, default=2,
                        help="Frames per second to extract (default: 2)")
    parser.add_argument("--max-frames", type=int, default=7200,
                        help="Maximum frames to extract (default: 7200)")
    args = parser.parse_args()

    print("=" * 60)
    print("Edge Video Preprocessing Demo - Data Preparation")
    print("=" * 60)
    print()

    # Phase 1: CIFAR-10
    print("[Phase 1] CIFAR-10 Dataset")
    print("-" * 40)
    if not download_cifar10():
        print("[ERROR] CIFAR-10 preparation failed.")
        return 1
    print()

    # Phase 2: Video frames
    print("[Phase 2] Video Frame Extraction")
    print("-" * 40)

    if args.video:
        success = extract_video_frames(args.video, args.fps, args.max_frames)
    else:
        # Check if frames already exist
        meta = parse_video_meta()
        if meta and meta.get("frame_count", 0) > 0:
            print("[Video] Frames already extracted ({} frames). Skipping.".format(
                meta["frame_count"]))
            success = True
        else:
            # Generate synthetic frames as fallback for demo
            success = create_synthetic_frames(args.max_frames)

    if not success:
        print("[WARNING] Video frame preparation had issues.")
        print("[WARNING] The training phase will still work with CIFAR-10 only.")
        print("[WARNING] The inference phase needs video_frames/ data.")
    print()

    print("=" * 60)
    print("Data preparation complete.")
    print("CIFAR-10 dataset: {}".format(DATASET_DIR))
    print("Video frames:     {}".format(VIDEO_DIR))
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
