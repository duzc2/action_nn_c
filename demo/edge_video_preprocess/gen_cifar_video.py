"""
gen_cifar_video.py — 用真实 CIFAR-10 图片生成可见运动序列 (快速版)
"""

import argparse, os, sys, time
from pathlib import Path
import numpy as np

SCRIPT_DIR = Path(__file__).parent.resolve()
DATASET_DIR = SCRIPT_DIR / "dataset"
OUTPUT_DIR = SCRIPT_DIR / "video_frames"
CIFAR_H, CIFAR_W, CIFAR_C = 32, 32, 3
FRAME_SIZE = CIFAR_H * CIFAR_W * CIFAR_C
CLASS_NAMES = ["airplane","automobile","bird","cat","deer",
               "dog","frog","horse","ship","truck"]


def load_cifar(filepath, max_n=1000):
    """向量化加载 CIFAR-10 二进制."""
    raw = np.fromfile(str(filepath), dtype=np.uint8)
    n = min(len(raw)//3073, max_n)
    raw = raw[:n*3073].reshape(n, 3073)
    labels = raw[:, 0].copy()
    r = raw[:, 1:1025].reshape(n, 32, 32).astype(np.float32) / 255.0
    g = raw[:, 1025:2049].reshape(n, 32, 32).astype(np.float32) / 255.0
    b = raw[:, 2049:3073].reshape(n, 32, 32).astype(np.float32) / 255.0
    return np.stack([r, g, b], axis=-1), labels


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", type=int, default=400)
    p.add_argument("--obj-size", type=int, default=14)
    p.add_argument("--motion-len", type=int, default=12)
    p.add_argument("--static-len", type=int, default=30)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    np.random.seed(args.seed)

    # 加载数据
    batch = DATASET_DIR / "data_batch_1.bin"
    if not batch.exists():
        print(f"[ERROR] Not found: {batch}"); sys.exit(1)

    print(f"Loading CIFAR-10 ...")
    t0 = time.time()
    images, labels = load_cifar(str(batch), max_n=500)
    print(f"  {len(images)} images in {time.time()-t0:.1f}s")

    # 选背景
    bg = images[42].copy()  # 固定背景

    # 选前景对象 (每个类别取 2 个, 裁剪中心 obj_size x obj_size)
    objs = []
    sz = args.obj_size
    cx0 = 16 - sz // 2
    for cat in range(10):
        idxs = np.where(labels == cat)[0]
        for k in range(min(2, len(idxs))):
            objs.append(images[idxs[k * len(idxs)//min(2,len(idxs))],
                         cx0:cx0+sz, cx0:cx0+sz, :].copy())

    print(f"  {len(objs)} foreground objects ({sz}x{sz})")

    # 生成帧
    n = args.frames
    mlen = args.motion_len
    slen = args.static_len
    period = mlen + slen
    frames = np.zeros((n, 32, 32, 3), dtype=np.float32)
    truth = np.zeros(n, dtype=np.int32)

    print(f"Generating {n} frames ...")
    t0 = time.time()
    seg = 0
    idx = 0
    while idx < n:
        fg = objs[seg % len(objs)]
        sx = np.random.randint(0, 33 - sz)
        sy = np.random.randint(0, 33 - sz)
        ex = np.random.randint(0, 33 - sz)
        ey = np.random.randint(0, 33 - sz)
        while abs(ex-sx)+abs(ey-sy) < 12:
            ex = np.random.randint(sz, 31 - sz)
            ey = np.random.randint(sz, 31 - sz)

        for i in range(min(period, n - idx)):
            fi = idx + i
            frame = np.clip(bg + np.random.uniform(-0.005, 0.005, bg.shape), 0.0, 1.0)
            if i < mlen:
                frac = i / max(mlen - 1, 1)
                cx = int(sx + (ex - sx) * frac)
                cy = int(sy + (ey - sy) * frac)
                truth[fi] = 1
            else:
                cx, cy = ex, ey
                truth[fi] = 0
            cx = max(0, min(cx, 32 - sz))
            cy = max(0, min(cy, 32 - sz))
            frame[cy:cy+sz, cx:cx+sz] = fg
            frames[fi] = frame

        idx += min(period, n - idx)
        seg += 1
        if seg % 5 == 0:
            print(f"  ... {idx}/{n} frames")

    print(f"  Done in {time.time()-t0:.1f}s")

    # 写入文件
    os.makedirs(str(OUTPUT_DIR), exist_ok=True)
    dat = OUTPUT_DIR / "video_frames.dat"
    meta = OUTPUT_DIR / "video_meta.txt"

    print(f"Writing {dat} ...")
    frames.astype(np.float32).tofile(str(dat))

    print(f"Writing {meta} ...")
    meta.write_text(f"frame_count {n}\nfps 2\nwidth 32\nheight 32\nchannels 3\n")

    mc = int(np.sum(truth))
    print(f"\nDone: {n} frames, {mc} motion ({mc/n*100:.0f}%), "
          f"{n-mc} static, {seg} segments")
    print(f"\nNow: python visualize_pipeline.py --max-frames {n}")


if __name__ == "__main__":
    main()
