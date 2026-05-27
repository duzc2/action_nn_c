"""
visualize_pipeline.py — Edge Video Preprocess 可视化工具
========================================================

读取 video_frames.dat 原始帧数据，用 Python 复现 C 端的运动检测管道
(SAD + 自适应阈值 + 时序平滑)，生成带标注的 MP4 视频和总结图表。

用法:
  python visualize_pipeline.py                          # 处理全部7200帧，生成视频+图表
  python visualize_pipeline.py --max-frames 500          # 只处理前500帧(快速预览)
  python visualize_pipeline.py --no-video                # 只生成图表，不生成视频
  python visualize_pipeline.py --start 1000 --end 1200   # 处理特定范围的帧

依赖:
  pip install numpy matplotlib pillow

输出:
  output/pipeline_video.mp4     — 逐帧标注视频
  output/sad_timeline.png       — SAD 全程时间线
  output/motion_stats.png        — 运动检测统计图表
  output/class_distribution.png  — 分类分布(需 CNN 输出数据)
"""

import argparse
import os
import struct
import sys
import time
from collections import defaultdict

import numpy as np

# ── 文件路径 ─────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_META  = os.path.join(SCRIPT_DIR, "video_frames", "video_meta.txt")
DEFAULT_FRAMES = os.path.join(SCRIPT_DIR, "video_frames", "video_frames.dat")
OUTPUT_DIR    = os.path.join(SCRIPT_DIR, "output")

VIDEO_W, VIDEO_H, VIDEO_C = 32, 32, 3
VIDEO_FRAME_SIZE = VIDEO_W * VIDEO_H * VIDEO_C  # 3072

# CIFAR-10 类别
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
CLASS_COLORS = [  # 每个类别一种颜色
    "#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6",
    "#1abc9c", "#e67e22", "#34495e", "#7f8c8d", "#c0392b",
]


# ═══════════════════════════════════════════════════════════════════════
#  数据加载
# ═══════════════════════════════════════════════════════════════════════

def load_video_meta(meta_path=DEFAULT_META):
    """读取 video_meta.txt, 返回 dict。"""
    meta = {}
    with open(meta_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                key = parts[0]
                val = parts[1]
                if "." in val:
                    meta[key] = float(val)
                else:
                    meta[key] = int(val)
    return meta


DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset")


def generate_frames_from_cifar(n_frames=300, obj_size=14,
                                motion_len=12, static_len=30, seed=42):
    """用 CIFAR-10 真实图片生成可见运动序列.

    背景: 随机 CIFAR-10 图像
    前景: 各类别裁剪的 obj_size x obj_size 对象, 在背景上移动
    运动期: 对象平移 → SAD 高
    静止期: 对象停留 → SAD 低

    返回: frames (N,32,32,3) float32, truth (N,) int32
    """
    import numpy as _np

    batch_file = os.path.join(DATASET_DIR, "data_batch_1.bin")
    if not os.path.exists(batch_file):
        raise FileNotFoundError(
            f"CIFAR-10 not found: {batch_file}\n"
            f"  Run: python data_prep.py")

    _np.random.seed(seed)
    raw = _np.fromfile(batch_file, dtype=_np.uint8)
    n_imgs = min(len(raw) // 3073, 500)
    raw = raw[:n_imgs * 3073].reshape(n_imgs, 3073)
    r = raw[:, 1:1025].reshape(n_imgs, 32, 32).astype(_np.float32) / 255.0
    g = raw[:, 1025:2049].reshape(n_imgs, 32, 32).astype(_np.float32) / 255.0
    b = raw[:, 2049:3073].reshape(n_imgs, 32, 32).astype(_np.float32) / 255.0
    imgs = _np.stack([r, g, b], axis=-1)
    labels = raw[:, 0]

    # 背景
    bg = imgs[seed % len(imgs)].copy()

    # 前景对象: 每个类别取 2 个, 裁剪 obj_size x obj_size
    objs = []
    sz = obj_size
    cx0 = 16 - sz // 2
    for cat in range(10):
        idxs = _np.where(labels == cat)[0]
        for k in range(min(2, len(idxs))):
            objs.append(imgs[int(idxs[k * len(idxs) // min(2, len(idxs))]),
                         cx0:cx0 + sz, cx0:cx0 + sz, :].copy())

    period = motion_len + static_len
    frames = _np.zeros((n_frames, 32, 32, 3), dtype=_np.float32)
    truth = _np.zeros(n_frames, dtype=_np.int32)

    idx = 0
    seg = 0
    while idx < n_frames:
        fg = objs[seg % len(objs)]
        sx = _np.random.randint(0, 33 - sz)
        sy = _np.random.randint(0, 33 - sz)
        ex = _np.random.randint(0, 33 - sz)
        ey = _np.random.randint(0, 33 - sz)
        while abs(ex - sx) + abs(ey - sy) < 12:
            ex = _np.random.randint(sz + 1, 31 - sz)
            ey = _np.random.randint(sz + 1, 31 - sz)

        for i in range(min(period, n_frames - idx)):
            fi = idx + i
            frame = _np.clip(bg + _np.random.uniform(-0.005, 0.005, bg.shape),
                            0.0, 1.0)
            if i < motion_len:
                frac = i / max(motion_len - 1, 1)
                cx = int(sx + (ex - sx) * frac)
                cy = int(sy + (ey - sy) * frac)
                truth[fi] = 1
            else:
                cx, cy = ex, ey
                truth[fi] = 0
            cx = max(0, min(cx, 32 - sz))
            cy = max(0, min(cy, 32 - sz))
            frame[cy:cy + sz, cx:cx + sz] = fg
            frames[fi] = frame

        idx += min(period, n_frames - idx)
        seg += 1

    return frames, truth


def load_video_frames(frames_path=DEFAULT_FRAMES, max_frames=None,
                      start=0):
    """读取 video_frames.dat, 返回 shape (N, 32, 32, 3) float32 np.ndarray。
       值域: [0.0, 1.0]。
    """
    file_size = os.path.getsize(frames_path)
    total_frames = file_size // (VIDEO_FRAME_SIZE * 4)
    if max_frames is not None:
        total_frames = min(total_frames, start + max_frames)

    frames = np.zeros((total_frames - start, VIDEO_H, VIDEO_W, VIDEO_C),
                      dtype=np.float32)

    with open(frames_path, "rb") as f:
        f.seek(start * VIDEO_FRAME_SIZE * 4)
        for i in range(total_frames - start):
            raw = f.read(VIDEO_FRAME_SIZE * 4)
            if len(raw) < VIDEO_FRAME_SIZE * 4:
                frames = frames[:i]
                break
            # 解析为 float32 (little-endian), 然后 reshape 为 (32, 32, 3)
            arr = np.frombuffer(raw, dtype=np.float32)
            # 数据是 interleaved RGB: R0,G0,B0,R1,G1,B1,...
            frames[i] = arr.reshape(VIDEO_H, VIDEO_W, VIDEO_C)

    return frames


# ═══════════════════════════════════════════════════════════════════════
#  运动检测管道 (Python 复现, 与 video_processor.c 逻辑一致)
# ═══════════════════════════════════════════════════════════════════════

class AdaptiveThreshold:
    """EMA-based 自适应阈值, 公式: threshold = base + factor * rolling_std"""

    def __init__(self, base=50.0, factor=2.0, alpha=0.1):
        self.base = base
        self.factor = factor
        self.alpha = alpha
        self.rolling_mean = 0.0
        self.rolling_std = 0.0
        self.initialized = False

    def update(self, sad):
        if not self.initialized:
            self.rolling_mean = sad
            self.rolling_std = 0.0
            self.initialized = True
        else:
            delta = sad - self.rolling_mean
            self.rolling_mean += self.alpha * delta
            abs_delta = abs(delta)
            self.rolling_std += self.alpha * (abs_delta - self.rolling_std)

    def get(self):
        if not self.initialized:
            return self.base
        return self.base + self.factor * self.rolling_std


class TemporalSmoother:
    """时序平滑器, 防抖动: 需要 N 个连续运动帧触发, M 个连续静止帧解除。"""

    def __init__(self, motion_confirm=3, static_confirm=10):
        self.motion_confirm = motion_confirm
        self.static_confirm = static_confirm
        self.consecutive_motion = 0
        self.consecutive_static = 0
        self.is_active = False

    def update(self, detected):
        if detected:
            self.consecutive_motion += 1
            self.consecutive_static = 0
            if not self.is_active and \
               self.consecutive_motion >= self.motion_confirm:
                self.is_active = True
        else:
            self.consecutive_static += 1
            self.consecutive_motion = 0
            if self.is_active and \
               self.consecutive_static >= self.static_confirm:
                self.is_active = False
        return self.is_active


def compute_sad(prev_frame, cur_frame):
    """Sum of Absolute Differences between two frames."""
    return float(np.sum(np.abs(cur_frame - prev_frame)))


def run_motion_pipeline(frames,
                        adapt_base=50.0, adapt_factor=2.0, ema_alpha=0.1,
                        motion_confirm=3, static_confirm=10,
                        cooldown_frames=0):
    """对全部帧运行运动检测管道, 返回每帧的诊断记录。

    Returns:
        records: list of dict, 每帧一条:
            frame_idx, sad, threshold, ema_std, raw_detect,
            debounced, is_active, should_infer
    """
    adap = AdaptiveThreshold(adapt_base, adapt_factor, ema_alpha)
    smoother = TemporalSmoother(motion_confirm, static_confirm)
    records = []
    cooldown_counter = 0

    for i, cur in enumerate(frames):
        rec = {"frame_idx": i, "sad": 0.0, "threshold": 0.0,
               "ema_std": 0.0, "raw_detect": 0, "debounced": 0,
               "is_active": 0, "should_infer": 0}

        if i == 0:
            # 第一帧: 没有 SAD, 但始终跑推理
            adap.update(0.0)
            rec["debounced"] = int(smoother.is_active)
            rec["should_infer"] = 1
        else:
            sad = compute_sad(frames[i - 1], cur)
            adap.update(sad)
            thr = adap.get()
            raw = 1 if sad > thr else 0
            deb = smoother.update(raw)

            rec["sad"] = sad
            rec["threshold"] = thr
            rec["ema_std"] = adap.rolling_std
            rec["raw_detect"] = raw
            rec["debounced"] = int(deb)
            rec["is_active"] = int(smoother.is_active)

            if cooldown_frames > 0:
                if deb and cooldown_counter == 0:
                    rec["should_infer"] = 1
                    cooldown_counter = cooldown_frames
                elif cooldown_counter > 0:
                    cooldown_counter -= 1
            else:
                rec["should_infer"] = int(deb)

        records.append(rec)

    return records


# ═══════════════════════════════════════════════════════════════════════
#  图表生成 (matplotlib)
# ═══════════════════════════════════════════════════════════════════════

def generate_sad_timeline(records, output_path):
    """生成 SAD 全程时间线图: 上半部分 SAD vs 阈值, 下半部分 运动状态热力图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    n = len(records)
    indices = np.arange(n)
    sad_vals = np.array([r["sad"] for r in records])
    thr_vals = np.array([r["threshold"] for r in records])
    raw_detect = np.array([r["raw_detect"] for r in records])
    debounced = np.array([r["debounced"] for r in records])

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(20, 8),
                                         sharex=True,
                                         gridspec_kw={"height_ratios": [3, 0.8, 0.8]})

    # ── 上: SAD 和阈值 ──
    ax1.plot(indices, sad_vals, alpha=0.5, color="#3498db", linewidth=0.5,
             label="SAD (sum of absolute differences)")
    ax1.plot(indices, thr_vals, color="#e74c3c", linewidth=1.0,
             label=f"Threshold (base=50 + 2*EMA_std)")

    # 标注运动事件
    in_event = False
    event_start = 0
    for i in range(n):
        if debounced[i] and not in_event:
            event_start = i
            in_event = True
        elif not debounced[i] and in_event:
            ax1.axvspan(event_start, i - 1, alpha=0.08, color="#2ecc71")
            in_event = False
    if in_event:
        ax1.axvspan(event_start, n - 1, alpha=0.08, color="#2ecc71")

    ax1.set_ylabel("SAD / Threshold", fontsize=11)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Motion Detection — SAD vs Adaptive Threshold", fontsize=13)

    # ── 中: raw_detect 热力图 ──
    ax2.fill_between(indices, 0, raw_detect, step="mid", color="#e67e22", alpha=0.7)
    ax2.set_ylabel("Raw\nDetect", fontsize=9)
    ax2.set_ylim(0, 1.2)
    ax2.set_yticks([0, 1])
    ax2.grid(True, alpha=0.3, axis="y")

    # ── 下: debounced 热力图 ──
    ax3.fill_between(indices, 0, debounced, step="mid", color="#2ecc71", alpha=0.7)
    ax3.set_ylabel("Debounced\nMotion", fontsize=9)
    ax3.set_xlabel("Frame index", fontsize=11)
    ax3.set_ylim(0, 1.2)
    ax3.set_yticks([0, 1])
    ax3.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] SAD 时间线 → {output_path}")


def generate_motion_stats_chart(records, output_path):
    """生成运动统计饼图 + 分布直方图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(records)
    motion_count = sum(1 for r in records if r["debounced"])
    static_count = n - motion_count
    raw_count = sum(1 for r in records if r["raw_detect"])
    total_sad = sum(r["sad"] for r in records)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 饼图: 运动 vs 静止帧
    ax = axes[0]
    ax.pie(
        [motion_count, static_count],
        labels=[f"Motion ({motion_count})",
                f"Static ({static_count})"],
        colors=["#e74c3c", "#2ecc71"],
        autopct="%1.1f%%",
        startangle=90,
    )
    ax.set_title(f"Frames: {n} total\nMotion: {motion_count} "
                 f"({motion_count/n*100:.1f}%)",
                 fontsize=11)

    # 柱状图: raw vs debounced 对比
    ax = axes[1]
    categories = ["Raw Detect\n(SAD > Thr)", "Debounced\n(Smoother)"]
    counts = [raw_count, motion_count]
    bars = ax.bar(categories, counts, color=["#e67e22", "#2ecc71"],
                  edgecolor="white", linewidth=1.5)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(counts) * 0.02,
                str(count), ha="center", fontsize=12, fontweight="bold")
    ax.set_ylabel("Frame Count", fontsize=11)
    ax.set_title(f"Raw vs Debounced\n"
                 f"Smoother filtered {raw_count - motion_count} false triggers",
                 fontsize=11)

    # 事件数统计
    ax = axes[2]
    events = []
    in_event = False
    for i in range(n):
        if records[i]["debounced"] and not in_event:
            events.append({"start": i})
            in_event = True
        elif not records[i]["debounced"] and in_event:
            events[-1]["end"] = i - 1
            in_event = False
    if in_event:
        events[-1]["end"] = n - 1

    durations = [e["end"] - e["start"] + 1 for e in events]
    if durations:
        ax.hist(durations, bins=20, color="#3498db", edgecolor="white",
                alpha=0.8)
        ax.axvline(x=np.mean(durations), color="#e74c3c", linestyle="--",
                   linewidth=2, label=f"Mean: {np.mean(durations):.0f} fr")
        ax.axvline(x=np.median(durations), color="#f39c12", linestyle="-.",
                   linewidth=2, label=f"Median: {np.median(durations):.0f} fr")
        ax.legend(fontsize=8)
        ax.set_xlabel("Event Duration (frames)", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(f"Motion Events: {len(events)} events\n"
                     f"Duration: mean={np.mean(durations):.0f}, "
                     f"max={max(durations)}, min={min(durations)}",
                     fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] 运动统计 → {output_path}")


# ═══════════════════════════════════════════════════════════════════════
#  视频生成 (matplotlib.animation)
# ═══════════════════════════════════════════════════════════════════════

def generate_dashboard_video(frames, records, output_path,
                             fps=10, upscale=8, class_preds=None):
    """生成带标注的 MP4 视频。

    每帧显示:
      - 中央: 32×32 帧图像(放大 upscale 倍)
      - 边框: 绿色=静态, 红色=运动触发中
      - 左上角: 帧号, SAD 值, 阈值
      - 左上角: 分类标签 + 置信度 (如有 CNN 输出)
      - 左下角: 运动状态指示器
      - 底部: 滚动 SAD 时间线(最近 N 帧)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    n_frames = len(frames)
    display_size = 32 * upscale  # 显示尺寸

    # 预计算 SAD 数组
    sad_arr = np.array([r["sad"] for r in records])
    thr_arr = np.array([r["threshold"] for r in records])
    deb_arr = np.array([r["debounced"] for r in records])

    # 算全局最大 SAD (用于底部图表缩放)
    sad_max = max(np.max(sad_arr[1:]), 1.0)

    # 创建图形
    fig = plt.figure(figsize=(12, 7), facecolor="#1a1a2e")
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 5, 2],
                          width_ratios=[1, 4, 1.5],
                          hspace=0.15, wspace=0.1)

    # ── 标题栏 ──
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.set_facecolor("#16213e")
    ax_title.set_xticks([])
    ax_title.set_yticks([])
    for spine in ax_title.spines.values():
        spine.set_visible(False)
    title_text = ax_title.text(
        0.5, 0.5,
        "Edge Video Preprocess — Motion Detection Pipeline",
        fontsize=14, fontweight="bold", color="white",
        ha="center", va="center",
        fontfamily="monospace",
    )

    # ── 帧显示区 ──
    ax_frame = fig.add_subplot(gs[1, 1])
    ax_frame.set_facecolor("#0f0f23")
    ax_frame.set_xticks([])
    ax_frame.set_yticks([])
    ax_frame.set_xlim(0, display_size)
    ax_frame.set_ylim(0, display_size)
    ax_frame.invert_yaxis()

    # 边框
    border = FancyBboxPatch(
        (0, 0), display_size, display_size,
        boxstyle="round,pad=4",
        linewidth=4, edgecolor="#2ecc71", facecolor="none",
    )
    ax_frame.add_patch(border)

    img_display = ax_frame.imshow(
        np.zeros((VIDEO_H, VIDEO_W, 3)),
        extent=[0, display_size, display_size, 0],
        aspect="auto",
    )

    # 帧号 / 状态文本框
    info_text = ax_frame.text(
        10, 10, "", fontsize=8, color="white",
        fontfamily="monospace", va="top", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.6),
    )

    # 运动指示器
    motion_indicator = ax_frame.text(
        display_size - 10, display_size - 10, "",
        fontsize=9, color="white", fontweight="bold",
        fontfamily="monospace", va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.4",
                  facecolor="black", alpha=0.6),
    )

    # ── 左侧统计面板 ──
    ax_stats = fig.add_subplot(gs[1, 0])
    ax_stats.set_facecolor("#16213e")
    ax_stats.set_xticks([])
    ax_stats.set_yticks([])
    for spine in ax_stats.spines.values():
        spine.set_visible(False)
    stats_panel = ax_stats.text(
        0.5, 0.5, "", fontsize=8, color="#a0a0b0",
        fontfamily="monospace", va="center", ha="center",
    )

    # ── 右侧分类面板 ──
    ax_class = fig.add_subplot(gs[1, 2])
    ax_class.set_facecolor("#16213e")
    ax_class.set_xticks([])
    ax_class.set_yticks([])
    for spine in ax_class.spines.values():
        spine.set_visible(False)
    class_panel = ax_class.text(
        0.5, 0.5, "", fontsize=8, color="#a0a0b0",
        fontfamily="monospace", va="center", ha="center",
    )

    # ── 底部 SAD 时间线 ──
    ax_timeline = fig.add_subplot(gs[2, :])
    ax_timeline.set_facecolor("#0f0f23")
    ax_timeline.set_xlabel("Frame", fontsize=9, color="#888899")
    ax_timeline.set_ylabel("SAD", fontsize=9, color="#888899")
    ax_timeline.tick_params(colors="#888899", labelsize=7)
    for spine in ax_timeline.spines.values():
        spine.set_color("#333355")

    # 预绘制两条线
    timeline_window = 200
    sad_line, = ax_timeline.plot([], [], color="#3498db", linewidth=1.0,
                                 alpha=0.8, label="SAD")
    thr_line, = ax_timeline.plot([], [], color="#e74c3c", linewidth=1.5,
                                 alpha=0.9, label="Threshold")
    ax_timeline.legend(fontsize=7, loc="upper right", framealpha=0.5)

    # 当前帧位置线
    cursor_line = ax_timeline.axvline(x=0, color="white", linewidth=1.0,
                                       alpha=0.5)

    # 运动高亮 (用一个持久化的引用容器, 在 update 中 remove+重建)
    motion_hl_container = {"patch": None}

    # 累计分类计数
    class_counts = defaultdict(int)
    total_inferred = 0

    def init():
        sad_line.set_data([], [])
        thr_line.set_data([], [])
        return [img_display, info_text, motion_indicator,
                border, title_text, sad_line, thr_line,
                cursor_line, stats_panel, class_panel]

    def update(frame_idx):
        nonlocal total_inferred
        rec = records[frame_idx]
        fr = frames[frame_idx]

        # ── 帧图像: clamp to [0, 1] ──
        disp = np.clip(fr, 0.0, 1.0)
        img_display.set_array(disp)

        # ── 边框颜色 (红=运动, 绿=静止) ──
        if rec["debounced"]:
            border.set_edgecolor("#e74c3c")
        elif rec["raw_detect"]:
            border.set_edgecolor("#e67e22")  # 橙=raw 检测到但未去抖
        else:
            border.set_edgecolor("#2ecc71")

        # ── 左上角信息 ──
        status_str = "MOTION" if rec["debounced"] else "STATIC"
        info_lines = [
            f"Frame: {frame_idx + 1} / {n_frames}",
            f"SAD:   {rec['sad']:8.1f}",
            f"Thr:   {rec['threshold']:8.1f}",
            f"Std:   {rec['ema_std']:8.1f}",
            f"Raw:   {rec['raw_detect']}   Deb: {rec['debounced']}",
            f"Status: {status_str}",
        ]
        info_text.set_text("\n".join(info_lines))

        # ── 右下角指示器 ──
        if rec["debounced"]:
            motion_indicator.set_text(" DETECTED ")
            motion_indicator.set_bbox(
                dict(boxstyle="round,pad=0.4",
                     facecolor="#e74c3c", alpha=0.85))
        elif rec["raw_detect"]:
            motion_indicator.set_text(" (confirming) ")
            motion_indicator.set_bbox(
                dict(boxstyle="round,pad=0.4",
                     facecolor="#e67e22", alpha=0.85))
        else:
            motion_indicator.set_text(" IDLE ")
            motion_indicator.set_bbox(
                dict(boxstyle="round,pad=0.4",
                     facecolor="#2ecc71", alpha=0.85))

        # ── 左侧统计 ──
        ti = frame_idx + 1
        md = sum(1 for r in records[:ti] if r["debounced"])
        sk = ti - md
        stats_lines = [
            " STATS ",
            "───────",
            f"Frames:  {ti:>5d}",
            f"Motion:  {md:>5d}",
            f"Skipped: {sk:>5d}",
            f"Ratio:   {md/ti*100:>4.1f}%",
            "",
            " SAVINGS ",
            "───────",
            f"Raw:   {ti*12.288:>6.0f} KB",
            f"Filter:{md*12.288:>6.0f} KB",
            f"Saved: {sk*12.288:>6.0f} KB",
        ]
        stats_panel.set_text("\n".join(stats_lines))

        # ── 右侧分类 ──
        # (此处预留 CNN 分类结果, 当前用随机模拟)
        if rec.get("pred_class") is not None:
            pred_class = rec["pred_class"]
            confidence = rec.get("confidence", 0.0)
            class_counts[pred_class] += 1
            total_inferred += 1

            lines = [" CNN (last) ",
                     "───────────",
                     f"Class: {CLASS_NAMES[pred_class]}",
                     f"Conf:  {confidence:.3f}",
                     "",
                     " DISTRIBUTION ",
                     "───────────"]
            if total_inferred > 0:
                top3 = sorted(class_counts.items(), key=lambda x: -x[1])[:5]
                for cls, cnt in top3:
                    pct = cnt / total_inferred * 100
                    lines.append(f"{CLASS_NAMES[cls]:>10s} {pct:4.1f}%")
            class_panel.set_text("\n".join(lines))
        else:
            class_panel.set_text(" CNN\n(inference\n results\n not\n loaded)")

        # ── 底部时间线 (最近 timeline_window 帧) ──
        start_i = max(0, frame_idx - timeline_window)
        end_i = min(n_frames, frame_idx + timeline_window // 4)
        x = np.arange(start_i, end_i)
        sad_line.set_data(x, sad_arr[start_i:end_i])
        thr_line.set_data(x, thr_arr[start_i:end_i])

        ax_timeline.set_xlim(start_i, end_i)
        sad_ymax = max(np.max(sad_arr[start_i:end_i]),
                       np.max(thr_arr[start_i:end_i]),
                       1.0)
        ax_timeline.set_ylim(0, sad_ymax * 1.1)

        # 运动高亮 (remove old + create new axvspan)
        if motion_hl_container["patch"] is not None:
            motion_hl_container["patch"].remove()
            motion_hl_container["patch"] = None

        highlight_start = None
        for ii in range(start_i, end_i):
            if deb_arr[ii] and highlight_start is None:
                highlight_start = ii
            elif not deb_arr[ii] and highlight_start is not None:
                motion_hl_container["patch"] = ax_timeline.axvspan(
                    highlight_start, ii - 1, alpha=0.1, color="#2ecc71")
                break
        if highlight_start is not None and motion_hl_container["patch"] is None:
            motion_hl_container["patch"] = ax_timeline.axvspan(
                highlight_start, end_i - 1, alpha=0.1, color="#2ecc71")

        # 当前帧光标
        cursor_line.set_xdata([frame_idx, frame_idx])

        # 更新标题
        total_md = sum(1 for r in records[:ti] if r["debounced"])
        title_text.set_text(
            f"Frame {frame_idx + 1:>5d} / {n_frames}  |  "
            f"Motion: {total_md:>5d} ({total_md/ti*100:4.1f}%)  |  "
            f"SAD: {rec['sad']:8.1f}  |  "
            f"Threshold: {rec['threshold']:8.1f}"
        )

        return [img_display, info_text, motion_indicator,
                border, title_text, sad_line, thr_line,
                cursor_line, stats_panel, class_panel]

    # 创建动画
    print(f"\n  Generating video: {n_frames} frames @ {fps} fps ...")
    t_start = time.time()

    ani = FuncAnimation(
        fig, update,
        frames=range(n_frames),
        init_func=init,
        blit=False,
        interval=1000 / fps,
    )

    writer = FFMpegWriter(
        fps=fps,
        metadata={"title": "Edge Video Preprocess Pipeline"},
        bitrate=2000,
    )
    ani.save(output_path, writer=writer, dpi=100)

    elapsed = time.time() - t_start
    print(f"  [OK] 视频 → {output_path} ({elapsed:.1f}s, "
          f"{n_frames/elapsed:.1f} fps render)")

    plt.close()


# ═══════════════════════════════════════════════════════════════════════
#  主入口
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Edge Video Preprocess 可视化工具")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="最大处理帧数 (0 = 全部)")
    parser.add_argument("--start", type=int, default=0,
                        help="起始帧索引")
    parser.add_argument("--end", type=int, default=0,
                        help="结束帧索引")
    parser.add_argument("--fps", type=int, default=10,
                        help="输出视频帧率 (默认 10)")
    parser.add_argument("--upscale", type=int, default=12,
                        help="帧图像放大倍数 (默认 12 -> 384x384)")
    parser.add_argument("--no-video", action="store_true",
                        help="只生成图表, 不生成视频")
    parser.add_argument("--adapt-base", type=float, default=50.0,
                        help="自适应阈值基数 (默认 50)")
    parser.add_argument("--adapt-factor", type=float, default=2.0,
                        help="自适应阈值因数 (默认 2.0)")
    parser.add_argument("--ema-alpha", type=float, default=0.1,
                        help="EMA 平滑系数 (默认 0.1)")
    parser.add_argument("--motion-confirm", type=int, default=3,
                        help="连续运动帧确认数 (默认 3)")
    parser.add_argument("--static-confirm", type=int, default=10,
                        help="连续静止帧确认数 (默认 10)")
    parser.add_argument("--use-cifar", action="store_true",
                        help="从 CIFAR-10 数据集生成真实图片运动序列")
    parser.add_argument("--obj-size", type=int, default=14,
                        help="CIFAR 模式下前景对象大小 (默认 14)")
    parser.add_argument("--motion-len", type=int, default=12,
                        help="CIFAR 模式下每段运动帧数 (默认 12)")
    parser.add_argument("--static-len", type=int, default=30,
                        help="CIFAR 模式下每段静止帧数 (默认 30)")
    parser.add_argument("--cifar-seed", type=int, default=42,
                        help="CIFAR 随机种子 (默认 42)")

    args = parser.parse_args()

    # 检查依赖
    try:
        import numpy as _numpy
        import matplotlib as _mpl
        from PIL import Image as _pil_image
    except ImportError as e:
        print(f"[ERROR] Missing dependency: {e}")
        print("Install: pip install numpy matplotlib pillow")
        sys.exit(1)

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if args.use_cifar:
        # ── CIFAR-10 模式: 用真实图片生成运动序列 ──
        n_frames = args.max_frames if args.max_frames > 0 else 300
        print(f"Generating {n_frames} frames from CIFAR-10 dataset ...")
        print(f"  Object: {args.obj_size}x{args.obj_size}, "
              f"motion={args.motion_len}fr, static={args.static_len}fr")
        t0 = time.time()
        frames, _cifar_truth = generate_frames_from_cifar(
            n_frames=n_frames, obj_size=args.obj_size,
            motion_len=args.motion_len, static_len=args.static_len,
            seed=args.cifar_seed)
        n_frames = len(frames)
        t1 = time.time()
        print(f"  Generated {n_frames} frames in {t1 - t0:.1f}s")
        if args.adapt_base == 50.0:
            # CIFAR 模式下默认降低阈值基数 (对象移动产生的 SAD 大约 30-80)
            args.adapt_base = 30.0
            print(f"  Auto-adjusted --adapt-base to 30 (optimized for CIFAR)")

        # 写入 video_frames.dat 供后续分析
        os.makedirs(os.path.join(SCRIPT_DIR, "video_frames"), exist_ok=True)
        np.array(frames, dtype=np.float32).tofile(
            os.path.join(SCRIPT_DIR, "video_frames", "video_frames.dat"))
        with open(os.path.join(SCRIPT_DIR, "video_frames", "video_meta.txt"), "w") as f:
            f.write(f"frame_count {n_frames}\nfps 2\nwidth 32\nheight 32\nchannels 3\n")
    else:
        # ── 文件模式: 读取已有的 video_frames.dat ──
        print("Loading metadata ...")
        meta = load_video_meta()
        total_frames = meta.get("frame_count", 7200)

        start = args.start
        end = args.end if args.end > 0 else total_frames
        if args.max_frames > 0:
            end = min(end, start + args.max_frames)
        n_frames = end - start

        print(f"Loading frames {start}..{end - 1} ({n_frames} frames) ...")
        t0 = time.time()
        frames = load_video_frames(max_frames=n_frames, start=start)
        n_frames = len(frames)
        t1 = time.time()
        print(f"  Loaded {n_frames} frames in {t1 - t0:.1f}s "
              f"({n_frames / (t1 - t0):.0f} fps)")

        if n_frames == 0:
            print("[ERROR] No frames loaded. Check video_frames.dat path.")
            sys.exit(1)

    # 运行运动检测
    print(f"\nRunning motion detection pipeline ...")
    print(f"  Parameters: base={args.adapt_base}, factor={args.adapt_factor}, "
          f"alpha={args.ema_alpha}")
    print(f"  Smoother: {args.motion_confirm} confirm / "
          f"{args.static_confirm} idle")
    t0 = time.time()
    records = run_motion_pipeline(
        frames,
        adapt_base=args.adapt_base,
        adapt_factor=args.adapt_factor,
        ema_alpha=args.ema_alpha,
        motion_confirm=args.motion_confirm,
        static_confirm=args.static_confirm,
    )
    t1 = time.time()

    motion_count = sum(1 for r in records if r["debounced"])
    static_count = sum(1 for r in records if not r["debounced"])
    print(f"  Processed in {t1 - t0:.1f}s")
    print(f"  Motion: {motion_count} ({motion_count/n_frames*100:.1f}%)")
    print(f"  Static: {static_count} ({static_count/n_frames*100:.1f}%)")

    # 生成图表
    print("\nGenerating charts ...")
    generate_sad_timeline(
        records,
        os.path.join(OUTPUT_DIR, "sad_timeline.png"),
    )
    generate_motion_stats_chart(
        records,
        os.path.join(OUTPUT_DIR, "motion_stats.png"),
    )

    # 生成视频
    if not args.no_video:
        video_path = os.path.join(OUTPUT_DIR, "pipeline_video.mp4")
        generate_dashboard_video(
            frames, records, video_path,
            fps=args.fps,
            upscale=args.upscale,
        )

    print(f"\nDone! Output files in: {OUTPUT_DIR}")
    print(f"  {os.path.join(OUTPUT_DIR, 'sad_timeline.png')}")
    print(f"  {os.path.join(OUTPUT_DIR, 'motion_stats.png')}")
    if not args.no_video:
        print(f"  {os.path.join(OUTPUT_DIR, 'pipeline_video.mp4')}")

    # 打印摘要
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"  {'─'*56}")
    print(f"  Total frames:     {n_frames:>6d}")
    print(f"  Motion detected:  {motion_count:>6d} ({motion_count/n_frames*100:5.1f}%)")
    print(f"  Static / skipped: {static_count:>6d} ({static_count/n_frames*100:5.1f}%)")
    print(f"  Bandwidth saved:  {static_count * 12.288 / 1024:>8.2f} MB")
    total_sad = sum(r["sad"] for r in records)
    avg_sad = total_sad / n_frames if n_frames > 1 else 0
    print(f"  Avg SAD:          {avg_sad:>8.1f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
